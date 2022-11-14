import torch
from torch import nn
from torch.nn import functional as F
from functorch import vmap 
import functools


class Loss(nn.Module):
    def __init__(self, full_weight, grad_weight, occ_prob_weight, elastic_weight, bg_weight):
        super().__init__()
        self.full_weight = full_weight
        self.grad_weight = grad_weight
        self.occ_prob_weight = occ_prob_weight
        self.elastic_loss_weight = elastic_weight
        self.bg_loss_weight = bg_weight
        self.l1_loss = nn.L1Loss(reduction='sum')
    
    def get_rgb_full_loss(self,rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1])
        return rgb_loss

    def get_smooth_loss(self, diff_norm):
        if diff_norm is None or diff_norm.shape[0]==0:
            return torch.tensor(0.0).cuda().float()
        else:
            return diff_norm.mean()

    def get_elastic_loss(self, jacobian, eps=1e-6, loss_type='log_svals'):
        """Compute the elastic regularization loss. FROM NERFIES 
        The loss is given by sum(log(S)^2). This penalizes the singular values
        when they deviate from the identity since log(1) = 0.0,
        where D is the diagonal matrix containing the singular values.
        Args:
            jacobian: the Jacobian of the point transformation.
            eps: a small value to prevent taking the log of zero.
            loss_type: which elastic loss type to use.
        Returns:
            The elastic regularization loss.
        """
        # if type(jacobian) is tuple:
        #     print('warning: did not do jacobian elastic loss right: line 41 n 70 in losses.py')
            # jacobian = jacobian[0]
        if loss_type == 'log_svals':
            svals = torch.linalg.svdvals(jacobian[0])
            log_svals = torch.log(torch.maximum(svals, torch.tensor(eps)))
            sq_residual = torch.sum(log_svals**2, axis=-1)
            # svals = torch.linalg.svdvals(jacobian[1])
            # log_svals = torch.log(torch.maximum(svals, torch.tensor(eps)))
            # sq_residual2 = torch.sum(log_svals**2, axis=-1)
        elif loss_type == 'svals':
            svals = torch.linalg.svd(jacobian, compute_uv=False)
            sq_residual = torch.sum((svals - 1.0)**2, axis=-1)
        elif loss_type == 'jtj':
            jtj = jacobian @ jacobian.T
            sq_residual = ((jtj - torch.eye(3)) ** 2).sum() / 4.0
        elif loss_type == 'div':
            div = torch.trace(jacobian, axis1=-2, axis2=-1) - 3.0
            sq_residual = div ** 2
        elif loss_type == 'det':
            det = torch.linalg.det(jacobian)
            sq_residual = (det - 1.0) ** 2
        elif loss_type == 'log_det':
            det = torch.linalg.det(jacobian)
            sq_residual = torch.log(torch.maximum(det, eps)) ** 2
        # elif loss_type == 'nr':
        #     rot = nearest_rotation_svd(jacobian)
        #     sq_residual = torch.sum((jacobian - rot) ** 2)
        else:
            raise NotImplementedError(
                f'Unknown elastic loss type {loss_type!r}')
        residual = torch.sqrt(sq_residual)
        loss = self.general_loss_with_squared_residual(sq_residual, alpha=-2.0, scale=0.03)
        # loss2 = self.general_loss_with_squared_residual(sq_residual2, alpha=-2.0, scale=0.03)
        return loss[0][0] #+ residual[0][0] ### SO BAD, originally loss, residual

    def get_background_loss(
        model, state, params, key, points, noise_std, alpha=-2, scale=0.001):
        raise(NotImplementedError)
        """Compute the background regularization loss."""
        metadata = random.choice(key,
                                jnp.array(model.warp_ids, jnp.uint32),
                                shape=(points.shape[0], 1))
        point_noise = noise_std * random.normal(key, points.shape)
        points = points + point_noise

        warp_field = model.create_warp_field(model, num_batch_dims=1)
        warp_out = warp_field.apply(
            {'params': params['warp_field']},
            points, metadata, state.warp_extra, False, False)
        warped_points = warp_out['warped_points'][..., :3]
        sq_residual = jnp.sum((warped_points - points)**2, axis=-1)
        loss = utils.general_loss_with_squared_residual(
            sq_residual, alpha=alpha, scale=scale)
        return loss

    def general_loss_with_squared_residual(self, squared_x, alpha, scale):
        r"""The general loss that takes a squared residual. FROM NERFIES
        This fuses the sqrt operation done to compute many residuals while preserving
        the square in the loss formulation.
        This implements the rho(x, \alpha, c) function described in "A General and
        Adaptive Robust Loss Function", Jonathan T. Barron,
        https://arxiv.org/abs/1701.03077.
        Args:
            squared_x: The residual for which the loss is being computed. x can have
            any shape, and alpha and scale will be broadcasted to match x's shape if
            necessary.
            alpha: The shape parameter of the loss (\alpha in the paper), where more
            negative values produce a loss with more robust behavior (outliers "cost"
            less), and more positive values produce a loss with less robust behavior
            (outliers are penalized more heavily). Alpha can be any value in
            [-infinity, infinity], but the gradient of the loss with respect to alpha
            is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
            interpolation between several discrete robust losses:
                alpha=-Infinity: Welsch/Leclerc Loss.
                alpha=-2: Geman-McClure loss.
                alpha=0: Cauchy/Lortentzian loss.
                alpha=1: Charbonnier/pseudo-Huber loss.
                alpha=2: L2 loss.
            scale: The scale parameter of the loss. When |x| < scale, the loss is an
            L2-like quadratic bowl, and when |x| > scale the loss function takes on a
            different shape according to alpha.
        Returns:
            The losses for each element of x, in the same shape as x.
        """
        eps = torch.finfo(torch.float32).eps
        eps = torch.tensor(eps).to('cuda')
        alpha = torch.tensor(alpha).to('cuda')

        # This will be used repeatedly.
        squared_scaled_x = squared_x / (scale ** 2)

        # The loss when alpha == 2.
        loss_two = 0.5 * squared_scaled_x
        # The loss when alpha == 0.
        loss_zero = torch.log1p(torch.minimum(0.5 * squared_scaled_x, torch.tensor(3e37)))
        # The loss when alpha == -infinity.
        loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
        # The loss when alpha == +infinity.
        loss_posinf = torch.expm1(torch.minimum(0.5 * squared_scaled_x, torch.tensor(87.5)))

        # The loss when not in one of the above special cases.
        # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
        beta_safe = torch.maximum(eps, torch.abs(alpha - 2.))
        # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
        alpha_safe = torch.where(
            torch.ge(alpha, 0.), torch.ones_like(alpha),
            -torch.ones_like(alpha)) * torch.maximum(eps, torch.abs(alpha))
        loss_otherwise = (beta_safe / alpha_safe) * (
            torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

        # Select which of the cases of the loss to return.
        loss = torch.where(
            alpha == -torch.inf, loss_neginf,
            torch.where(
                alpha == 0, loss_zero,
                torch.where(
                    alpha == 2, loss_two,
                    torch.where(alpha == torch.inf, loss_posinf, loss_otherwise))))

        return scale * loss

    # def nearest_rotation_svd(matrix, eps=1e-6):
    #     """Computes the nearest rotation using SVD."""
    #     # TODO(keunhong): Currently this produces NaNs for some reason.
    #     u, _, vh = torch.linalg.svd(matrix + eps, compute_uv=True, full_matrices=False)
    #     # Handle the case when there is a flip.
    #     # M will be the identity matrix except when det(UV^T) = -1
    #     # in which case the last diagonal of M will be -1.
    #     det = torch.linalg.det(u @ vh)
    #     m = torch.stack([torch.ones_like(det), torch.ones_like(det), det], axis=-1)
    #     m = torch.diag(m)
    #     r = u @ m @ vh
    #     return 

    def forward(self, rgb_pred, rgb_gt, diff_norm, jacobian_mat, bg_points=None, elastic_loss_type='log_svals'):
        rgb_gt = rgb_gt.cuda()
        
        if self.full_weight != 0.0:
            rgb_full_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt)
        else:
            rgb_full_loss = torch.tensor(0.0).cuda().float()

        if diff_norm is not None and self.grad_weight != 0.0:
            grad_loss = self.get_smooth_loss(diff_norm)
        else:
            grad_loss = torch.tensor(0.0).cuda().float()

        if jacobian_mat is not None and self.elastic_loss_weight != 0.0:
            # elastic_fn = functools.partial(self.get_elastic_loss,
            #                                 loss_type=elastic_loss_type)
            # v_elastic_fn = vmap(vmap(elastic_fn))
            # elastic_loss = v_elastic_fn(jacobian_mat)
            elastic_loss = self.get_elastic_loss(jacobian_mat)
        else:
            elastic_loss = torch.tensor(0.0).cuda().float()

        if bg_points is not None and self.bg_loss_weight != 0.0:
            bg_loss = self.get_background_loss(bg_points)
        else: 
            bg_loss = torch.tensor(0.0).cuda().float()

        loss = self.full_weight * rgb_full_loss + \
               self.grad_weight * grad_loss + \
                self.elastic_loss_weight * elastic_loss + \
                self.bg_loss_weight * bg_loss
        if torch.isnan(loss):
            breakpoint()

        return {
            'loss': loss,
            'fullrgb_loss': rgb_full_loss,
            'grad_loss': grad_loss,
            'elastic_loss': elastic_loss,
            'background_loss': bg_loss
        }


