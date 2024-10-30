import numpy as np
import torch
from typing import List
from rich import print

__all__ = [
    "humanise_feature_extractor",
    "get_humanise_motion_feature_dim_list",
    "get_humanise_level_idx",
]

# define the constant
num_joints = 22 # number SMPL joint
feature_dim = num_joints * 3
parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19], dtype=np.int32)

def jointsidx_to_humaniseidx(idx_list: List[int]):
    joints_feature_idx = []
    for joint_index in idx_list:
        joints_feature_idx.extend(range(3 * (joint_index), 3 * (joint_index + 1)))
    return joints_feature_idx



## ----- version 1 -----
# level 0: pelvis
# level 1: pelvis, torso
# level 2: pelvis, torso, left/right leg
# level 3: pelvis, torso, left/right leg, left/right arm
# level 4: all joints (pelvis, torso, left/right leg, left/right arm, head)



def get_humanise_motion_feature_dim_list(num_level, humanise_level_idx):    
    feature_dim_list = [len(humanise_level_idx[i]) for i in range(len(humanise_level_idx))]
    residual = num_level - len(humanise_level_idx)
    feature_dim_list.extend([feature_dim_list[-1]] * residual)
    return feature_dim_list

def humanise_feature_extractor(data, level, humanise_level_idx):
    if level < len(humanise_level_idx):
        return data[:, :, humanise_level_idx[level]]
    else:
        return data[:, :, humanise_level_idx[-1]]

def get_humanise_level_idx(num_level):
    
    level_0_joints_idx = [0]
    level_1_joints_idx = [0, 3, 6, 9]
    level_2_joints_idx = [0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11]
    level_3_joints_idx = [0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21]

    
    humanise_level_idx = [
        jointsidx_to_humaniseidx(level_0_joints_idx),
        jointsidx_to_humaniseidx(level_1_joints_idx),
        jointsidx_to_humaniseidx(level_2_joints_idx),
        jointsidx_to_humaniseidx(level_3_joints_idx),
        list(range(feature_dim)),
    ]
    if num_level >= len(humanise_level_idx):
        return humanise_level_idx
    else:
        return humanise_level_idx[-num_level:]