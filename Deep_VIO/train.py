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
from os import listdir
import os

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

def loss_function(q_pred, t_pred, q_true, t_true):
    
    #translation loss
    criterion_t = torch.nn.MSELoss()
    loss_t = criterion_t(t_true, t_pred)
    
    #quaternion loss
    criterion_q = torch.sum(q_true * q_pred, dim=1)
    q_norm_true = torch.norm(q_true, p=2, dim=1)
    q_norm_pred = torch.norm(q_pred, p=2, dim=1)
    loss_q = 1-torch.abs(criterion_q/(q_norm_true*q_norm_pred))
    loss_q = torch.mean(loss_q)
    
    # merging the translation and quaternion loss, as translation loss is usually much larger than quaternion loss, multiplying it with a factor
    beta = 10
    loss = loss_t + beta*loss_q
    
    return loss

def BatchGenVIO(images, IMU, pose, minibatch):
    #generate batches of data
    batches = []
    total_batches = int(len(images)/minibatch)
    for i in range(total_batches):
        batched_images = []
        batched_IMU = []
        batched_pos_q = []
        batched_pos_t = []
        random_index = np.random.choice(len(images)-1, minibatch, replace=False)
        for idx in random_index:
            #prepping images
            current_image = images[idx] #taking 2 subsequent images and merging them
            next_image = images[idx+1]
            merged_image = np.stack((current_image, next_image), axis=2)
            torch_merged_image = torch.from_numpy(merged_image).float() # converting it to torch tensor
            torch_merged_image = torch_merged_image.view(2, 800, 800) #hard coding number of pixels here, need to change that based on pixel size
            batched_images.append(torch_merged_image.unsqueeze(0))  # adding unsqueeze to add batch dimension to the tensor
            
            #prepping IMU
            current_IMU = IMU[idx*10:(idx+1)*10, 1:] #taking 100 IMU values (not taking timestamps)
            torch_current_IMU = torch.from_numpy(current_IMU).float() # converting it to torch tensor
            torch_current_IMU = torch_current_IMU.reshape(10,6) # reshaping it to 6,100
            batched_IMU.append(torch_current_IMU.unsqueeze(0))
            
            #prepping Rotations(labels)
            current_pose_t = pose[idx*10, 1:4] #taking the T values from the pose
            current_pose_q = pose[idx*10, 4:] #taking quaternions
            torch_current_pose_t = torch.from_numpy(current_pose_t).float() # converting it to torch tensor
            torch_current_pose_q = torch.from_numpy(current_pose_q).float() # converting it to torch tensor
            batched_pos_t.append(torch_current_pose_t.unsqueeze(0))
            batched_pos_q.append(torch_current_pose_q.unsqueeze(0))
        
        batched_images = torch.cat(batched_images, dim=0) #concatenating all the images in the batch
        batched_IMU = torch.cat(batched_IMU, dim=0) #concatenating all the IMU values in the batch
        
        batched_pos_t = torch.cat(batched_pos_t, dim=0) #concatenating all the T values in the batch
        batched_pos_q = torch.cat(batched_pos_q, dim=0) #concatenating all the quaternion values in the batch
        
        batch = [batched_images, batched_IMU, batched_pos_t, batched_pos_q]
        batches.append(batch) #appending the batch to the list of batches
        
    return batches

def TrainOperation(Batches, Epochs, lr, latestmodelpath, checkpointpath, logspath, device, modeltype):
    """_summary_

    Args:
        Batches (toorch list): _input generated batches
        epochs (float): for hoow many epochs the model should run
        lr (Float): learning rate in optimizer
        latestmodelpath (str): if we want to continue training from a saved model
        checkpointpath (str):where the checkpoints should be saved
        logspath (str): where the tensorlogs should be saved
        device (str): device to run the model on
    """
    
    total_batches = len(Batches)
    lossvalues_eachepoch = []
    
    #initialize model
    if modeltype == "vionet":
        model = VIoNet(input_channels=2) # 2 input channels
        model = model.to(device)
    
    if modeltype == "ionet":
        model = IoNet(6, 64, 2, 4) # 2 input channels
        model = model.to(device)
        
    if modeltype == "vonet":
        model = VoNet(2) # 2 input channels
        model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print("Model Initialized")
    
    if latestmodelpath is not None:
        checkpoint = torch.load(latestmodelpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    print("training started...")
    
    if not os.path.exists(logspath):
        os.makedirs(logspath)
    writer = SummaryWriter(log_dir=logspath)
        
    for epochs in tqdm(range(Epochs)):
        epoch_counter = epochs+1
        cumulative_Loss = 0
        for batch in Batches:
            
            #get the batch
            if modeltype == "vonet":
                images = batch[0].to(device)
            if modeltype == "ionet":
                IMU = batch[1].to(device)
                
            if modeltype == "vionet":
                images = batch[0].to(device)
                IMU = batch[1].to(device)
            
            pos_t = batch[2].to(device)
            pos_q = batch[3].to(device)
            
            #forward pass
            if modeltype == "vonet":
                pred_q, pred_t = model(images)
            if modeltype == "ionet":
                pred_q, pred_t = model(IMU)
            if modeltype == "vionet":
                pred_q, pred_t = model(images, IMU)
            
            
            #backward pass
            optimizer.zero_grad()
            
            loss = loss_function(pred_q, pred_t, pos_q, pos_t)
            loss.backward()
            optimizer.step() 
            
            cumulative_Loss = cumulative_Loss+loss
        
        epoch_avg_loss = cumulative_Loss/total_batches
        
        print(f"Epoch: {epochs}, Loss: {epoch_avg_loss}")
        
        writer.add_scalar('training loss',
                              epoch_avg_loss,
                              epochs)
        
        
        #saving the model checkpoint
        savename = checkpointpath + "checkpoint_"+ modeltype + str(epochs) + ".ckpt"
        if epoch_counter%20 == 0:
            torch.save({'epoch': Epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_avg_loss},
                        savename)
            print('\n' + savename + ' Model Saved...')
        
        #saving the loss values in an array for plotting
        lossvalues_eachepoch.append(epoch_avg_loss)
        
    return lossvalues_eachepoch

def plot_loss(Loss):
    Loss = [l.item() for l in Loss]
    plt.plot(np.arange(len(Loss)), Loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend()
    plt.show()
            
    
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to the images folder", default=r"data_a\eight\images")
    parser.add_argument("--IMU", type=str, help="path to imu data", default=r"data_a\eight\IMU.csv")
    parser.add_argument("--rotations", type=str, help="path to relative pose data", default=r"data_a\eight\abs_pose.csv")
    parser.add_argument("--minibatch", type=int, help="Minibatch size", default=1000)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=500)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--latestmodelpath", type=str, help="folder to load checkpoint and start training from that epoch")
    parser.add_argument("--checkpointpath", type=str, help="where should the checkpoints save", default="checkpoints/")
    parser.add_argument("--logspath", type=str, help="where the tensorlogs should save", default="tensor_logs/")
    parser.add_argument("--model", type=str, help="which model to run", default="ionet")
    args = parser.parse_args()

    #loading data
    images = LoadImagesFromFolder(args.images)
    IMU = pd.read_csv(args.IMU)
    IMU_array = IMU.values # IMU with time values
    pose = pd.read_csv(args.rotations)
    pose_array = pose.values
    minibatch = args.minibatch # rotation with time values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #generating batches
    Batches = BatchGenVIO(images, IMU_array, pose_array, minibatch)
    
    #train operation
    Loss = TrainOperation(Batches, args.epochs, args.lr, args.latestmodelpath, args.checkpointpath, args.logspath, device, args.model)
    
    #plotting losses
    plot_loss(Loss)
    
    
    
if __name__ == "__main__":
    main()