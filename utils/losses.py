import torch
import torch.nn as nn

class MultiScaleReConsLossWithMask(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(MultiScaleReConsLossWithMask, self).__init__()
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.nb_joints = nb_joints
        
    def forward(self, motion_pred, motion_gt, mask):
        mask = mask.to(torch.float32).unsqueeze(-1)
        loss = self.Loss(motion_pred, motion_gt)
        loss = (loss * (1. - mask)).sum((-2, -1)) / (1. - mask).sum((-2, -1)) / motion_gt.shape[-1]
        return torch.mean(loss)