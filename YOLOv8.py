import torch 
import torch.nn as nn
from torchinfo import summary

#YOLO Version 8
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class SiLU(nn.Module): # 
    @staticmethod
    def forward(x):
         return x * torch.sigmoid(x)

class Conv(nn.Module): 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False) 
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
    def forward (self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module) : 
    def __init__(self, c1, c2 ,shortcut=True, k=3, e=0.5):
        super().__init__()
        c_ = int(e*c2)
        self.cv1 = Conv(c1, c_, k, s=1, p=1)
        self.cv2 = Conv(c_, c2, k, s=1, p=1)
        self.add = shortcut and c1 == c2    
        
    def forward (self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  

class C2f(nn.Module): 
    def __init__(self, c1, c2, n=2, shortcut=False,g=1, e=0.5):
        super().__init__()
        self.c = int(c2*e)
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, k=3, e = 1.0) for _ in range(n))
        self.cv2 = Conv((2+n)*self.c, c2, 1)
        
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))
        
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

class Backbone(nn.Module):
    def __init__(self, base_channels, base_deepth, deep_mul, phi, pretrained = False):
        super().__init__()
        self.stem = Conv(3,base_channels, 3,2)
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels*2, 3,2),
            C2f(base_channels*2, base_channels*2, base_deepth, True)
        )
        
        self.dark3 = nn.Sequential (
           Conv(base_channels*2,base_channels*4,3,2),
           C2f(base_channels*4, base_channels*4, base_deepth*2, True) 
        )
        
        self.dark4 = nn.Sequential(
            Conv(base_channels*4, base_channels*8, 3, 2),
            C2f(base_channels*8, base_channels*8, base_deepth*2, True)
        )
        self.dark5 = nn.Sequential(
            # kali deep_mul karna Untuk mengatur jumlah channel dan jumlah blok sesuai ukuran model (n/s/m/l/x).
            Conv(base_channels*8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_deepth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), 5)
        )
        
    def forward(self,x):
        x= self.stem(x)
        x= self.dark2(x)
        x= self.dark3(x)
        feature1 = x
        x = self.dark4(x)
        feature2 = x
        x = self.dark5(x)
        feature3 = x
        return  feature1, feature2, feature3
    
backbone = Backbone(base_channels=32, base_deepth=3, deep_mul=1.0, phi=None)

# input dummy
x = torch.randn(1, 3, 640, 640)

# forward
f1, f2, f3 = backbone(x)

print("feature1:", f1.shape)
print("feature2:", f2.shape)
print("feature3:", f3.shape)

total_params = sum(p.numel() for p in backbone.parameters())
trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

print("Total params:", total_params)
print("Trainable params:", trainable_params)

for name, module in backbone.named_modules():
    if hasattr(module, "weight") or hasattr(module, "bias"):
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:<30} params: {params}")

summary(backbone, input_size=(1, 3, 640, 640))
# class Neck(nn.Module):
#     def __init__(self,): 
#         super().__init__()
        
#     def forward(self,x):
#         return  
    
# class Head(nn.Module):
#     def __init__(self,):
#         super().__init__()
      
    
#     def forward(self,x):
#         return  
        
    

