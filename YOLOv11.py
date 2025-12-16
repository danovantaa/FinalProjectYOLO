import torch 
import torch.nn as nn

# YOLO Version 11 

class SiLU(nn.Module): 
    @staticmethod
    def forward(x):
         return x * torch.sigmoid(x)

class Conv(nn.Module): #
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k,s, p )
        self.bn = nn.BatchNorm2d(c2)
        self.act = SiLU() # if act else nn.Identity()
        
    def forward (self, x):
        return self.act(self.bn(self.conv(x)))
    
class Bottleneck(nn.Module): 
    def __init__(self, c1, c2 ,shortcut=True, k=3, e=0.5):
        super().__init__()
        c_ = int(e*c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 2, 1)
        self.add = shortcut and c1 == c2  
        
    def forward (self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  

# class C3K(nn.Module):
#     def __init__(self, c1, c2, n=2, shortcut=False,g=1 ):
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.m = nn.ModuleList(Bottleneck(c2, c2, shortcut, g, k=3, e = 1.0) for _ in range(n))
#         self.cv2 = Conv((n+1)*c2, c2, 1,1)
        
#     def forward(self, x):
#         y = [self.cv1(x)]
#         for m in self.m:
#             y.append(m(y[-1]))
#         return self.cv2(torch.cat(y, dim=1))
 
# class C3K2(nn.Module):
#     def __init__(self,c1,c2, n, shortcut, g ):
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 2,1)
#         self.c = nn.ModuleList(C3K(c2, c2, n, shortcut, g) for _ in range(n))
#         self.cv2 = Conv((n+1)*c2, c2, 1,1 ) 
        
#     def forward(self, x):
#         y = [self.cv1(x)]
#         for c in self.c:
#             y.append(c(y[-1]))
#         return self.cv2(torch.cat(y, dim=1))

class C3K2(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=False):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(c2, c2, shortcut)for _ in range(n))
        self.cv2 = Conv(2 * c2, c2, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = y1
        for m in self.m:
            y2 = m(y2)
        return self.cv2(torch.cat((y1, y2), dim=1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, s=1):
        super().__init__()
        c_ = c1//2
        self.cv1 =Conv(c1, c_,1,1 ) 
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k//2)
        self.cv2 = Conv(c_*4, c2, 1,1) 
        
    def forward(self, x):
        x = self.cv1(x)
        max1 = self.m(x)
        max2 = self.m(max1)
        return self.cv2(torch.cat((x, max1, max2, self.m(max2)),1))

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_ratio = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

class PSA(nn.Module):
    def __init__(self, c, attn_ratio = 0.5, num_heads = 4, shortcut= True) -> None:
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1), 
            Conv(c * 2, c, 1, act=False)
            )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):
    def __init__(self, c1, c2, n = 1, e = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(
            *(PSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Backbone (nn.Module) : 
    def __init__(self, base_channels, base_deepth, deep_mul, phi, pretrained = False):
        super().__init__()
        # stage default n
        n2 = round(2 * deep_mul)   
        n3 = round(2 * deep_mul)   
        n4 = round(3 * deep_mul)   
        n5 = round(1 * deep_mul)
        
        self.stem = Conv(3, base_channels, 3,2)
        self.stage2 = nn.Sequential (
            Conv(base_channels, base_channels*2, 3,2),
            C3K2(base_channels*2, base_channels*2, n2, shortcut=True, g=1 )
        )
        self.stage3 = nn.Sequential (
            Conv(base_channels*2, base_channels*4, 3,2),
            C3K2(base_channels*4, base_channels*4, n3, shortcut=True, g=1 )
        )
        self.stage4 = nn.Sequential (
            Conv(base_channels*4, base_channels*8, 3,2),
            C3K2(base_channels*8, base_channels*8, n4, shortcut=True, g=1 )
        )
        self.stage5 = nn.Sequential (
            Conv(base_channels*8, base_channels*16, 3,2),
            #tidak dikalikan dengan deep_mul karna desain modul C3K2 dan C2PSA yang sudah dinormalisasi channel-nya dan tidak memerlukan scaling lebar kanal.
            C3K2(base_channels*16, base_channels*16,n5, shortcut=True, g=1  ),
            SPPF(base_channels * 16, base_channels * 16),
            C2PSA()
        )
        
    def forward(self, x):
        x= self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feature1 =x 
        x= self.stage4(x)
        feature2 =x 
        x=self.stage5(x)
        feature3 = x
        return feature1, feature2, feature3

backbone = Backbone(base_channels=32, base_deepth=3, deep_mul=1.0, phi=None)


class Neck (nn.Module): 
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        
        return 
    
class Head(nn.Module) :
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        
        return 