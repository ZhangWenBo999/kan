# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# from eca_module import *
# from MultiScaleFeatureFusionModule import *


# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#             )
#         self.pool_types = pool_types
#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( avg_pool )
#             elif pool_type=='max':
#                 max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( max_pool )
#             elif pool_type=='lp':
#                 lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( lp_pool )
#             elif pool_type=='lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp( lse_pool )

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw

#         scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale

# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs

# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = F.sigmoid(x_out) # broadcasting
#         return x * scale

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#         self.out = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, dilation=1)
#     def forward(self, x):
#         # x_out = self.ChannelGate(x)
#         x_out = eca(x)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
#         x_out = self.out(x_out)
#         return x_out

# if __name__ == '__main__':
#     x = torch.randn(4, 64, 32, 32)
#     eca = eca_layer(64)
#     cbam = CBAM(64)
#     MSFF = MultiScaleFeatureFusion(64, 64)
#     x = MSFF(x)
#     out = cbam(x)

#     print(out.shape)


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat

# 论文：Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.
# 论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

# Adaptive Sparse Self-Attention (ASSA)
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads=8, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2  # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    # Instantiate the WindowAttention_sparse class
    dim = 64  # Dimension of input features
    win_size = (64, 64)  # Window size(H, W)
    # Create an instance of the WindowAttention_sparse module
    window_attention_sparse = WindowAttention_sparse(dim, win_size)
    C = dim
    input = torch.randn(1, 64 * 64, C)#输入B H W
    # Forward pass
    output = window_attention_sparse(input)

    # Print input and output size
    print(input.size())
    print(output.size())

# import torch
# import torch.nn as nn
# from einops import rearrange
# import math

# # 论文：Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.
# # 论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
# # 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
# class FRFN(nn.Module):
#     def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
#         super().__init__()
#         self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
#                                      act_layer())
#         self.dwconv = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
#             act_layer())
#         self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
#         self.dim = dim
#         self.hidden_dim = hidden_dim

#         self.dim_conv = self.dim // 4
#         self.dim_untouched = self.dim - self.dim_conv
#         self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

#     def forward(self, x):
#         x_init = x
#         # bs x hw x c
#         bs, hw, c = x.size()
#         hh = int(math.sqrt(hw))

#         # spatial restore
#         x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

#         x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)

#         # flaten
#         x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

#         x = self.linear1(x)
#         # gate mechanism
#         x_1, x_2 = x.chunk(2, dim=-1)

#         x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
#         x_1 = self.dwconv(x_1)
#         x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
#         x = x_1 * x_2

#         x = self.linear2(x)

#         return x + x_init


# if __name__ == '__main__':
#     # Instantiate the FRFN class
#     dim = 64  # Dimension of input features


#     # Create an instance of the FRFN module
#     frfn = FRFN(dim)

#     # Generate a random input tensor
#     B = 1  # Batch size
#     H = 64  # Height of the feature map
#     W = 64  # Width of the feature map
#     C = dim  # Number of channels

#     input = torch.randn(B, H * W, C)

#     # Forward pass
#     output = frfn(input)

#     # Print input and output shapes
#     print(input.size())
#     print(output.size())