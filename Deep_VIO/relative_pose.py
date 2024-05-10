import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import pandas as pd
import csv

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


def get_relative_pose_quaternion(pose1, pose2):
    # Convert euler angles to rotation matrix
    R1 = convert_R_to_rotmat(pose1[3:])
    R2 = convert_R_to_rotmat(pose2[3:])

    T1 = np.array(pose1[:3])
    T2 = np.array(pose2[:3])

    # Get relative pose - Relative Rotation and Relative Translation
    relative_rot = np.dot(R1.T, R2)
    relative_pose = np.zeros(7)
    relative_pose[:3] = np.dot(R1.T, T2 - T1)
    relative_pose[3:] = convert_euler_to_quaternion(convert_rotmat_to_R(relative_rot))

    return relative_pose

def get_relative_pose_quaternion_T(pose1, pose2):
    # Convert euler angles to rotation matrix
    R1 = convert_R_to_rotmat(convert_quaternion_to_R(pose1[3:]))
    R2 = convert_R_to_rotmat(convert_quaternion_to_R(pose2[3:]))

    t1 = np.array(pose1[:3])
    t2 = np.array(pose2[:3])
    
    T1 = np.zeros((4, 4))
    T1[:3, :3] = R1
    T1[:3, 3] = t1
    T1[3, 3] = 1
    
    T2 = np.zeros((4, 4))
    T2[:3, :3] = R2
    T2[:3, 3] = t2
    T2[3, 3] = 1
    
    T2_inv = np.linalg.inv(T1)
    
    T_rel = np.dot(T2_inv, T2)
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    
    relative_pose = np.zeros(7)
    quaternions = convert_euler_to_quaternion(convert_rotmat_to_R(R_rel))
    relative_pose[:3] = t_rel
    relative_pose[3:] = quaternions    
    return relative_pose


abs_pose = pd.read_csv("data_a/eight/abs_pose.csv")
abs_pose = abs_pose.values

rel_pose = []

for i in range(1, len(abs_pose)):
    rel_pose.append(get_relative_pose_quaternion_T(abs_pose[i-1,1:], abs_pose[i,1:]))
    
rel_pose = np.array(rel_pose)

index = np.arange(1, len(rel_pose)+1).reshape(-1, 1)

headers = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

data_with_index = np.hstack([index, rel_pose])

csv_file = "data_a/eight/pose_rel.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data_with_index)

