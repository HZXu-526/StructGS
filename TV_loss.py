import torch
import torch.nn as nn
from torch.autograd import Variable
 
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

 