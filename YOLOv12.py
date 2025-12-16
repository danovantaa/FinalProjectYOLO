import torch 
import torch.nn as nn


# YOLO Version 12
class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
         return x * torch.sigmoid(x)

class Conv(nn.Module): #
    def __init__(self, c1, c2, k=3, s=2, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k,s, p )
        self.bn = nn.BatchNorm2d(c2)
        self.act = SiLU()
        
    def forward (self, x):
        return self.act(self.bn(self.conv(x)))  

# class AAttn(nn.Module):
#     def __init__(self, dim, num_heads, area=1):
#         super().__init__()
#         self.area = area

#         self.num_heads = num_heads
#         self.head_dim = head_dim = dim // num_heads
#         all_head_dim = head_dim * self.num_heads

#         self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
#         self.v = Conv(dim, all_head_dim, 1, act=False)
#         self.proj = Conv(all_head_dim, dim, 1, act=False)

#         self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


#     def forward(self, x):
#         B, C, H, W = x.shape
#         N = H * W

#         qk = self.qk(x).flatten(2).transpose(1, 2)
#         v = self.v(x)
#         pp = self.pe(v)
#         v = v.flatten(2).transpose(1, 2)

#         if self.area > 1:
#             qk = qk.reshape(B * self.area, N // self.area, C * 2)
#             v = v.reshape(B * self.area, N // self.area, C)
#             B, N, _ = qk.shape
#         q, k = qk.split([C, C], dim=2)

#         if x.is_cuda and USE_FLASH_ATTN:
#             q = q.view(B, N, self.num_heads, self.head_dim)
#             k = k.view(B, N, self.num_heads, self.head_dim)
#             v = v.view(B, N, self.num_heads, self.head_dim)

#             x = flash_attn_func(
#                 q.contiguous().half(),
#                 k.contiguous().half(),
#                 v.contiguous().half()
#             ).to(q.dtype)
#         else:
#             q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
#             k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
#             v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

#             attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
#             max_attn = attn.max(dim=-1, keepdim=True).values
#             exp_attn = torch.exp(attn - max_attn)
#             attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
#             x = (v @ attn.transpose(-2, -1))
#             x = x.permute(0, 3, 1, 2)

#         if self.area > 1:
#             x = x.reshape(B // self.area, N * self.area, C)
#             B, N, _ = x.shape
#         x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

#         return self.proj(x + pp)


class Backbone (nn.Module) : 
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        
        return 
    
class Neck (nn.Module): 
    def __init__(self, ):
        super().__init__()  
        
    def forward(self, x):
        
        return 
    
class Head(nn.Module) :
    def __init__(self, x):
        super().__init__()
        
        
    def forward(self, x):
        
        return 