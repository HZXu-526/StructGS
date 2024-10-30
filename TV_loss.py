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
        count_h = self._tensor_size(x[:,:,1:,:])  # Calculate the total number of differences computed
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # x[:,:,1:,:]-x[:,:,:h_x-1,:] performs a misalignment on the original image, creating two images where pixel positions differ by 1.
        # The first image starts from pixel 1 (the original starts from 0) and goes to the last pixel, while the second image starts from pixel 0
        # and ends at the second to last pixel. This operation misaligns the original image into two separate images. The subtraction that follows
        # calculates the difference between each pixel and its next adjacent pixel in the original image.
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

 