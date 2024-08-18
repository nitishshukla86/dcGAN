import torch
import torch.nn as nn

class CrossRoad(nn.Module):
    def __init__(self,cross=True):
        super(CrossRoad, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.cross=cross
    def forward(self, pred1, pred2, target1, target2):
        """
        Computes the sum of L1 loss between two pairs of images.

        Args:
            pred1 (torch.Tensor): The first predicted image.
            pred2 (torch.Tensor): The second predicted image.
            target1 (torch.Tensor): The first ground truth image.
            target2 (torch.Tensor): The second ground truth image.

        Returns:
            torch.Tensor: The sum of L1 losses.
        """
        loss1 = self.l1_loss(pred1, target1)
        loss2 = self.l1_loss(pred2, target2)
        cross1=self.l1_loss(pred1,target2)
        cross2=self.l1_loss(pred2,target1)
        hit= 0 if loss1+loss2<cross1+cross2 else 1
            

        
        total_loss = torch.min(loss1 + loss2, cross1+cross2) if self.cross else loss1 + loss2
        return total_loss,hit
