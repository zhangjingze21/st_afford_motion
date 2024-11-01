import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    # pred_id = torch.argmax(pred, dim=1)
    # mask = labels.ne(ignore_index)
    # n_correct = pred_id.eq(labels).masked_select(mask)
    # acc = torch.mean(n_correct.float()).item()
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc

def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) # torch.Size([24, 210, 1025]) torch.Size([24, 210])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss
