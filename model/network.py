import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as AF
from functorch import jacfwd, vmap
from model import types
from model import rigid_body as rigid
from typing import Any, Iterable, Optional, Dict, Tuple

class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)

class NeuralNetwork(nn.Module):
    ''' Network class containing occupancy and appearance (and warp) field
    
    Args:
        cfg (dict): network configs
    '''

    def __init__(self, cfg, **kwargs):
        super().__init__()
        out_dim = 4
        dim = 3
        self.num_layers = cfg['num_layers']
        hidden_size = cfg['hidden_dim']
        self.octaves_pe = cfg['octaves_pe']
        self.octaves_pe_views = cfg['octaves_pe_views']
        dim_code_warp = 1
        self.skips = cfg['skips']
        self.rescale = cfg['rescale']
        self.feat_size = cfg['feat_size']
        geometric_init = cfg['geometric_init'] 
        self.warp = cfg['warp'] 
        self.hyperwarp = cfg['hyperwarp'] 
        self.img_idxs = cfg['img_idxs']
        self.ambient_dim = cfg['ambient_dim']
        self.use_pivot = cfg['use_pivot']
        self.use_translation = cfg['use_translation']
        self.use_jac_condition = cfg['use_jac_condition']
        self.cond_app = cfg['condition_appearance']
        bias = 0.6

        if not self.hyperwarp:
            self.ambient_dim = 0

        # init positional encoding
        dim_warp  = dim_code_warp*self.octaves_pe*2 + dim + dim_code_warp 
        dim_embed = (dim + self.ambient_dim)*self.octaves_pe*2 + (dim + self.ambient_dim) + self.use_jac_condition
        dim_embed_view = ((dim + self.ambient_dim) + dim*self.octaves_pe_views*2 + dim + dim + self.feat_size
                             + self.cond_app *(dim_code_warp*self.octaves_pe*2 + dim_code_warp))
        self.transform_points = PositionalEncoding(L=self.octaves_pe)
        self.transform_points_view = PositionalEncoding(L=self.octaves_pe_views)
        self.octaves_pe_warp = cfg['octaves_pe_warp']
        
        ## warp network
        if self.warp is not None:
            if self.warp == 'translation':
                dims_warp = [dim_warp] + [ hidden_size for i in range(0, 4)] + [dim]
                self.warp_field = self.TranslationField(dims_warp)
            elif self.warp == 'SE3Field':
                self.skips_se3 = (4,)
                self.warp_field = self.SE3Field(dim_warp, dim)
            else:
                raise ValueError(f'Unknown warp field type: {self.warp!r}')
        
        if self.hyperwarp:
            self.skips_hyperwarp = (4,)
            dims_hyperwarp = [dim_warp] + [ 64 for i in range(0, 6)] + [self.ambient_dim]
            self.warp_field = self.HyperSheetMLP(dims_hyperwarp)

        ### geo network
        dims_geo = [dim_embed] + [ hidden_size if i in self.skips else hidden_size for i in range(0, self.num_layers)] + [self.feat_size+1] 
        self.num_layers = len(dims_geo)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
            else:
                out_dim = dims_geo[l + 1]
            lin = nn.Linear(dims_geo[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.octaves_pe > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_geo[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)


        ## appearance network
        dims_view = [dim_embed_view] + [ hidden_size for i in range(0, 4)] + [3]
        self.num_layers_app = len(dims_view)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def infer_occ(self, p, img_idx=None):
        p, img_idx = self.map_points(p, img_idx=img_idx)
        pe = self.transform_points(p/self.rescale)
        
        if self.use_jac_condition:
            assert(img_idx is not None)
            if len(img_idx.shape) == 3:
                img_idx = img_idx.reshape(-1, 1)
            jac = self.jacobian(p, img_idx)
            j = torch.linalg.norm(jac, dim=(-2,-1))
            pe = torch.cat([pe, j], dim=-1)

        x = pe

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skips:
                x = torch.cat([x, pe], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)     
        return x
    
    def infer_app(self, points, normals, view_dirs, feature_vectors, img_idx=None):
        points, img_idx = self.map_points(points, img_idx=img_idx)
        if self.cond_app:
            pe = self.transform_points_warp(img_idx)
            rendering_input = torch.cat([points, view_dirs, normals.squeeze(-2), feature_vectors, pe], dim=-1)
        else:
            rendering_input = torch.cat([points, view_dirs, normals.squeeze(-2), feature_vectors], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers_app - 1):
            lina = getattr(self, "lina" + str(l))
            x = lina(x)
            if l < self.num_layers_app - 2:
                x = self.relu(x)
        x = self.tanh(x) * 0.5 + 0.5
        return x
    
    def infer_warp(self, p, img_idx):
        if self.warp == 'translation':
            return self.translation_warp(p, img_idx)
        elif self.warp == 'SE3Field':
            return self.se3_warp(p, img_idx)
        else:
            raise ValueError(f'Unknown warp field type: {self.warp!r}')

    def gradient(self, p, img_idx=None):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.infer_occ(p, img_idx)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)

    def jacobian(self, p, img_idx):
        with torch.enable_grad():
            p.requires_grad_(True)
            img_idx = img_idx.to(torch.float32)
            if len(img_idx) == 1:
                img_idx = torch.ones_like(torch.unsqueeze(p[...,0],dim=-1))*img_idx
            img_idx.requires_grad_(True)     
            return vmap(jacfwd(self.infer_warp))(p,img_idx) 

    def warp_bg_points(self, points, noise_std=0.001, img_idx=None):
        if img_idx is not None:
            idx = torch.ones_like(torch.unsqueeze(points[...,0],dim=-1))*img_idx
        else:
            idx = torch.randint(len(self.img_idxs), size=(points.shape[0], 1))
            idx = self.img_idxs[idx]
        point_noise = torch.normal(torch.zeros(points.shape),noise_std)
        points = points + point_noise.to(points.device)
        return self.infer_warp(points.type(torch.float32), idx.type(torch.float32))

    def forward(self, p, ray_d=None, img_idx=None, only_occupancy=False, return_logits=False,return_addocc=False, noise=False, **kwargs):
        x = self.infer_occ(p, img_idx)
        if only_occupancy:
            return self.sigmoid(x[...,:1] * -10.0)
        elif ray_d is not None:
            
            input_views = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            input_views = self.transform_points_view(input_views)
            normals =  self.gradient(p, img_idx)
            #normals = n / (torch.norm(n, dim=-1, keepdim=True)+1e-6)
            rgb = self.infer_app(p, normals, input_views, x[...,1:], img_idx=img_idx)
            if return_addocc:
                if noise:
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
                else: 
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
            else:
                return rgb
        elif return_logits:
            return -1*x[...,:1]
        
    def map_points(self, p, img_idx=None):
        if self.warp is not None:
            assert(img_idx is not None)
            if len(img_idx) == 1:
                img_idx = torch.ones_like(torch.unsqueeze(p[...,0],dim=-1))*img_idx
            p  = self.infer_warp(p, img_idx=img_idx)
            if p.isnan().sum():
                ValueError('warp contains NaN values!!!')
        if self.hyperwarp:
            assert(img_idx is not None)
            if len(img_idx) == 1:
                img_idx = torch.ones_like(torch.unsqueeze(p[...,0],dim=-1))*img_idx
            ph = self.infer_hyperwarp(p, img_idx=img_idx)
            if ph.isnan().sum():
                ValueError('hyperwarp contains NaN values!!!')
            else:
                p = torch.cat([p, ph], dim=-1)
        return p, img_idx

    def TranslationField(self, dims_warp):
        self.num_layers_warp = len(dims_warp)
        self.transform_points_warp = PositionalEncoding(L=self.octaves_pe_warp)
        self.create_mlp(name='linw', dims=dims_warp, output_init='uniform')

    def translation_warp(self, p, img_idx):
        pe = self.transform_points_warp(img_idx)
        x = torch.cat([p, pe], dim=-1)
        for l in range(0, self.num_layers_warp - 1):
            linw = getattr(self, "linw" + str(l))
            x = linw(x)
            if l < self.num_layers_warp - 2:
                x = self.relu(x)
        return p + x
    
    def create_mlp(self,
                   name='lin',
                   dims=[],
                   output_init=None,
                   skips=[]):
        for l in range(0, len(dims) - 1):
            if l + 1 in skips:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            # initialize weights
            lin = nn.utils.weight_norm(lin)
            setattr(self, name + str(l), lin)
        if output_init == 'uniform':
            torch.nn.init.uniform_(lin.bias, -1e-4, 1e-4)
            torch.nn.init.uniform_(lin.weight, -1e-4, 1e-4)
        return

    def SE3Field(self, dim_warp, dim):
        self.transform_points_warp = PositionalEncoding(L=self.octaves_pe_warp)
        
        dims_trunk = [dim_warp] + [ 128 for i in range(0, 6)] 
        self.num_layers_trunk = len(dims_trunk)
        self.create_mlp(name='trunk', dims=dims_trunk, skips=self.skips_se3)
        
        dims_w = [ 128 ] + [dim]
        self.num_layers_w = len(dims_w)
        self.create_mlp(name='w', dims=dims_w, output_init='uniform')
        
        dims_v = [ 128 ] + [dim]
        self.num_layers_v = len(dims_v)
        self.create_mlp(name='v', dims=dims_v, output_init='uniform')

    def se3_warp(self, p, img_idx):
        if len(p) == 0:
            return p
        pe = self.transform_points_warp(img_idx)
        pe = pe.reshape(p.shape[:-1]+pe.shape[-1:])
        inputs = torch.cat([p, pe], dim=-1)
        x = inputs

        # trunk_output = self.trunk(inputs)
        for l in range(0, self.num_layers_trunk - 1):
            lin = getattr(self, "trunk" + str(l))
            if l in self.skips_se3:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers_trunk - 2:
                x = self.softplus(x) #DO I NEED THIS ??
        trunk_output = x
        
        # w = self.branches['w'](trunk_output)
        for l in range(0, self.num_layers_w - 1):
            lin = getattr(self, "w" + str(l))
            x = lin(x)
            if l < self.num_layers_w - 2:
                x = self.relu(x)
        w = x

        # v = self.branches['v'](trunk_output)
        x = trunk_output
        for l in range(0, self.num_layers_v - 1):
            lin = getattr(self, "v" + str(l))
            x = lin(x)
            if l < self.num_layers_v - 2:
                x = self.relu(x)
        v = x

        theta = torch.linalg.norm(w, dim=-1, keepdim=True)
        w = w / theta
        v = v / theta

        screw_axis = torch.cat([w, v], dim=-1)
        if len(screw_axis.shape) == 1 or screw_axis.shape[0] == 0.:
            N = 1
            B = 1
        elif len(screw_axis.shape) == 2:
            N,_ = screw_axis.shape
            B = 1
        else:
            B,N,_ = screw_axis.shape
        exp_se3 = vmap(rigid.exp_se3)
        transform = exp_se3(screw_axis.reshape(B*N,-1), theta.reshape(B*N,-1)).reshape(B,N,4,4)

        warped_points = p

        if self.use_pivot:
            raise(NotImplementedError)
            pivot = self.branches['p'](trunk_output)
            warped_points = warped_points + pivot

        warped_points = rigid.from_homogenous(
            torch.bmm(transform.reshape(B*N,4,4), 
            rigid.to_homogenous(warped_points).reshape(B*N,4,1)).reshape(B*N,4))

        if self.use_pivot:
            raise(NotImplementedError)
            warped_points = warped_points - pivot

        if self.use_translation:
            raise(NotImplementedError)
            t = self.branches['t'](trunk_output)
            warped_points = warped_points + t

        return warped_points
    
    def HyperSheetMLP(self, dims_hyperwarp):
        self.num_layers_hyperwarp = len(dims_hyperwarp)
        self.transform_points_hyperwarp = PositionalEncoding(L=self.octaves_pe_warp)
        self.create_mlp(name='linh', dims=dims_hyperwarp, output_init='uniform', skips=self.skips_hyperwarp)

    def infer_hyperwarp(self, p, img_idx):
        pe = self.transform_points_hyperwarp(img_idx)
        x = torch.cat([p, pe.reshape(p.shape[:-1]+pe.shape[-1:])], dim=-1)
        inputs = x
        for l in range(0, self.num_layers_hyperwarp - 1):
            linw = getattr(self, "linh" + str(l))
            if l in self.skips_hyperwarp:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
            x = linw(x)
            if l < self.num_layers_hyperwarp - 2:
                x = self.relu(x)
        return x
