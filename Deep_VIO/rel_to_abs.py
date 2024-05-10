import torch
import pandas as pd
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def convert_to_euler(rotation):
    return (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2]),
    )


def convert_euler_to_quaternion(rotation):
    r = R.from_euler("xyz", rotation, degrees=False)
    return r.as_quat()


def convert_R_to_rotmat(rotation):
    r = R.from_euler("xyz", rotation, degrees=False)
    return r.as_matrix()


def convert_quaternion_to_R(quaternion):
    r = R.from_quat(quaternion)
    return r.as_euler("xyz", degrees=False)


def convert_rotmat_to_R(rotmat):
    r = R.from_matrix(rotmat)
    return r.as_euler("xyz", degrees=False)


def relative_to_absolutepose(relative_pose):
    
    
    absolute_pose = torch.zeros_like(relative_pose)
    absolute_pose = absolute_pose.to(relative_pose.device)
    absolute_pose[0] = relative_pose[0]
    for i in tqdm(range(1, len(relative_pose))):
        
        d_translation = relative_pose[i, :3]
        d_rotation_q = relative_pose[i, 3:]
        
        #q values for previous frame
        q_prev = absolute_pose[i-1, 3:]
        
        d_rotation_matrix = q_to_rotation_matrix(q_prev)
        
        #accumulate the translation
        absolute_pose[i, :3] = absolute_pose[i-1, :3] + torch.matmul(d_rotation_matrix, d_translation)
        #accumulate the rotation
        absolute_pose[i, 3:] = quaternion_product(q_prev, d_rotation_q)
        
        absolute_pose[i, 3:] = normalize_quaternion(absolute_pose[i, 3:])
    
    return absolute_pose

def relative_to_absolutepose_T(relative_pose, absolute_pose_gt):
    
    absolute_pose = np.zeros_like(relative_pose)
    row_of_zeros = np.zeros((1, absolute_pose.shape[1]), dtype=absolute_pose.dtype)
    absolute_pose = np.concatenate((absolute_pose, row_of_zeros), axis=0)
    
    absolute_pose[0] = absolute_pose_gt[0,:]
    
    for i in range(1, len(relative_pose)):
        d_translation = relative_pose[i, :3]
        d_rotation_q = relative_pose[i, 3:]
        
        q_prev = absolute_pose[i-1, 3:]
        
        rot_mat = convert_R_to_rotmat(convert_quaternion_to_R(d_rotation_q))
        rot_mat_prev = convert_R_to_rotmat(convert_quaternion_to_R(q_prev))
        
        
        
        absolute_pose[i, :3] = absolute_pose[i, :3] + np.dot(rot_mat_prev, d_translation)
        rot_mat_abs = np.dot(rot_mat_prev, rot_mat)
        absolute_pose[i, 3:] = convert_euler_to_quaternion(convert_rotmat_to_R(rot_mat_abs))
    
    return absolute_pose
        
    
def q_to_rotation_matrix(q):
    
    x,y,z,w = q
    rotation_matrix = torch.tensor([[1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
                                    [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2,   2*y*z - 2*x*w],
                                    [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]])
    return rotation_matrix

def quaternion_product(q1, q2):
    
    #given 2 quaternions q1 and q2, transforms second quaternion in first quaternion's frame
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.tensor([x, y, z, w])

def normalize_quaternion(q):
    
    return q/torch.norm(q)

def save_to_txt(absolute_pose, filename):
    
    with open(filename, 'w') as f:
        f.write("# Timestamp tx ty tz qx qy qz qw\n")
        
        for i,pose in enumerate(absolute_pose):
            line = f"{i} {pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f}\n"
            f.write(line)
            
            
            
Pose_pred = pd.read_csv('abvenoutput.csv', sep=' ')
Pose_pred = Pose_pred.values
Pose_gt = pd.read_csv('abveninput.csv', sep=' ')
Pose_gt = Pose_gt.values
pose_abs_gt = pd.read_csv('cam.csv')
pose_abs_gt = pose_abs_gt.values

Pose_pred_u = Pose_pred[:,1:8]
Pose_gt_u = Pose_gt[:,1:8]

abs_pred = relative_to_absolutepose_T(Pose_pred_u, pose_abs_gt)
abs_gt = relative_to_absolutepose_T(Pose_gt_u, pose_abs_gt)

save_to_txt(abs_pred, 'abven_pred.txt')
save_to_txt(abs_gt, 'abven_gt.txt')
            
