"""Compact U-Net baseline, trained from scratch at the original image resolution."""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_VERSION='tiny_unet_v1'


class Block(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.ReLU(inplace=True),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self,input_channels=5,base=16):
        super().__init__()
        if base%4: raise ValueError('Base width must be divisible by four')
        self.down=nn.ModuleList([Block(input_channels,base),Block(base,base*2),Block(base*2,base*4)])
        self.center=Block(base*4,base*8)
        self.up=nn.ModuleList([Block(base*12,base*4),Block(base*6,base*2),Block(base*3,base)])
        self.output=nn.Conv2d(base,3,1)

    def forward(self,x):
        skips=[]
        for block in self.down:
            x=block(x);skips.append(x);x=F.max_pool2d(x,2)
        x=self.center(x)
        for block,skip in zip(self.up,reversed(skips)):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat([x,skip],dim=1))
        return self.output(x)


def segmentation_loss(logits,target,class_weights):
    ce=F.cross_entropy(logits.float(),target,weight=class_weights)
    probs=logits.float().softmax(1)
    truth=F.one_hot(target,3).permute(0,3,1,2).float()
    intersection=(probs*truth).sum((0,2,3)); total=probs.sum((0,2,3))+truth.sum((0,2,3))
    dice=(2*intersection+1)/(total+1)
    return ce+1-dice[1:].mean()
