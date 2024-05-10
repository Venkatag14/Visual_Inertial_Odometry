import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW
import cv2
import numpy as np
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import argparse
import math as m
from tqdm import tqdm
from os import listdir
from network import VoNet, VIoNet, IoNet
import pandas as pd
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def LoadImagesFromFolder(folder):
    images = []
    
    # Get a list of files in the folder and sort them based on the numeric part of the file name
    files = sorted(listdir(folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Load images
    for file in files:
        # Read the image
        tmp = cv2.imread(folder + "\\" + file)
        if tmp is not None:
            # Convert the image to black and white
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            images.append(tmp)
    return images

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

def BatchgenIMU(IMU, minibatch):
    #generate batches of IMU to send them to the network
    batches = []
    total_batches = int(len(IMU)/(minibatch*100))
    for i in range(total_batches):
        batched_IMU = []
        index = 0
        for idx in range (minibatch):
            current_IMU = current_IMU = IMU[(idx+index)*10:(idx+index+1)*10, 1:] #taking 100 IMU values (not taking timestamps)
            torch_current_IMU = torch.from_numpy(current_IMU).float() # converting it to torch tensor
            torch_current_IMU = torch_current_IMU.reshape(10,6) # reshaping it to 6,100
            batched_IMU.append(torch_current_IMU.unsqueeze(0))
        
        index += minibatch
        batched_IMU = torch.cat(batched_IMU, dim=0)
        batches.append(batched_IMU)
    
    return batches

def BatchGenVIO(images, IMU, minibatch):
    #generate batches of data
    batches = []
    total_batches = int(len(images)/minibatch)
    for i in range(total_batches-1):
        #prepping images
        current_image = images[i] #taking 2 subsequent images and merging them
        next_image = images[i+1]
        merged_image = np.stack((current_image, next_image), axis=2)
        torch_merged_image = torch.from_numpy(merged_image).float() # converting it to torch tensor
        torch_merged_image = torch_merged_image.view(2, 800, 800) #hard coding number of pixels here, need to change that based on pixel size  # adding unsqueeze to add batch dimension to the tensor
        
        #prepping IMU
        current_IMU = IMU[i*10:(i+1)*10, 1:] #taking 100 IMU values (not taking timestamps)
        torch_current_IMU = torch.from_numpy(current_IMU).float() # converting it to torch tensor
        torch_current_IMU = torch_current_IMU.reshape(10,6) # reshaping it to 6,100
        
        batch = [torch_merged_image, torch_current_IMU]
        batches.append(batch) #appending the batch to the list of batches
        
    return batches

def test(batches, Checkpointpath, device, modeltype):
    
    #initialize model and load the checkpoint
    if modeltype == "vionet":
        model = VIoNet()
    if modeltype == "ionet":
        model = IoNet()
    model = model.to(device)
    Checkpoint = torch.load(Checkpointpath)
    model.load_state_dict(Checkpoint['model_state_dict'])
    model.eval()
    
    #initializing q, t
    q = []
    t = []
    if modeltype == "ionet":
        #inferencing model
        with torch.no_grad():
            for batch in tqdm(batches):
                image = batch[0].to(device)
                image = image.unsqueeze(0)
                imu = batch[1].to(device)
                imu = imu.unsqueeze(0)
                q_pred, t_pred = model(imu)
                torch.cuda.empty_cache()
                q_pred = q_pred.to('cpu')
                t_pred = t_pred.to('cpu')

                q.append(q_pred)
                t.append(t_pred)
    
    if modeltype == "vionet":
        with torch.no_grad():
            for batch in tqdm(batches):
                image = batch[0].to(device)
                image = image.unsqueeze(0)
                imu = batch[1].to(device)
                imu = imu.unsqueeze(0)
                q_pred, t_pred = model(image, imu)
                torch.cuda.empty_cache()

                q_pred = q_pred.to('cpu')
                t_pred = t_pred.to('cpu')

                q.append(q_pred)
                t.append(t_pred)
        
    q = torch.cat(q, dim=0)
    t = torch.cat(t, dim=0)
    
    relative_pose = torch.cat((t, q), dim=1)
    
    return relative_pose

def relative_to_absolutepose(relative_pose, ground_truth):
    
    
    ground_truth = torch.tensor(ground_truth)
    absolute_pose = torch.zeros_like(relative_pose)
    absolute_pose = absolute_pose.to(relative_pose.device)
    row_of_zeros = torch.zeros(1, absolute_pose.size(1), dtype=absolute_pose.dtype)
    absolute_pose = torch.cat((absolute_pose, row_of_zeros), dim=0)
    
    absolute_pose[0] = ground_truth[0, 1:]
    for i in range(1, len(relative_pose)):
        
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
    
    absolute_pose[0] = absolute_pose_gt[0,1:]
    
    for i in range(1, len(relative_pose)):
        d_translation = relative_pose[i, :3]
        d_rotation_q = relative_pose[i, 3:]
        
        q_prev = absolute_pose[i-1, 3:]
        
        rot_mat = convert_R_to_rotmat(convert_quaternion_to_R(d_rotation_q))
        rot_mat_prev = convert_R_to_rotmat(convert_quaternion_to_R(q_prev))
        
        
        
        absolute_pose[i, :3] = absolute_pose[i-1, :3] + np.dot(rot_mat_prev, d_translation)
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
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to the images folder", default=r"data_a\eight\images")
    parser.add_argument("--IMU", type=str, help="path to imu data", default=r"data_a\eight\IMU.csv")
    parser.add_argument("--rotations", type=str, help="path to relative pose data", default=r"data_t\relative_pose.csv")
    parser.add_argument("--groundtruth", type=str, help="path to ground truth data", default=r"data_a\eight\abs_pose.csv")
    parser.add_argument("--minibatch", type=int, help="Minibatch size", default=1)
    parser.add_argument("--latestmodelpath", type=str, help="folder to load checkpoint", default=r"checkpoints\checkpoint_vionet39.ckpt")
    parser.add_argument("--model", type=str, help="which model to run", default="vionet")
    args = parser.parse_args()

    IMU = pd.read_csv(args.IMU)
    IMU_array = IMU.values
    
    images = LoadImagesFromFolder(args.images)
    
    batch_size = args.minibatch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #ground truth data
    gt_traj = pd.read_csv(args.groundtruth)
    gt_traj = gt_traj.values
    
    batches = BatchGenVIO(images, IMU_array, batch_size)
    
    modeltype = args.model
    
    
    relative_pose = test(batches, args.latestmodelpath, device, modeltype)
    
    relative_pose = relative_pose.cpu().detach().numpy()
    
    #absolute_pose = relative_to_absolutepose_T(relative_pose, gt_traj)
    
    save_to_txt(relative_pose, 'absolute_pose_pred_vio_new.txt')
    
    
    
    
    
if __name__ == "__main__":
    main()
