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
    
def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq
