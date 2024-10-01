# %% [markdown]
# 

# %%
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F

import time

# Record the start time
start_time = time.time()


def rgb_to_gray_image_conversion(filelist):

    rgb_image_list = []
    gray_image_list = []



    for file in filelist:
        image_rgb = cv2.imread(file)
        
        rgb_image_list.append(image_rgb)
        
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        
        gray_image_list.append(gray_image)
        

    return gray_image_list,rgb_image_list  


def file_to_gray(gray_image_list,index_no):

    
    gray_image = gray_image_list[index_no]
    gray_image = torch.tensor(gray_image).float()
    
    padded_gray_image = F.pad(gray_image, (1, 1, 1, 1), 'constant', 0)
    
    
    return padded_gray_image.detach().numpy() 




# filelist = glob.glob('2000/*.png')
# filelist.sort()

# print(filelist)


#gray_image_list,rgb_image_list = rgb_to_gray_image_conversion(filelist)

# print(file_to_gray(gray_image_list,0).shape)



# %%
import torch
import torch.nn.functional as F

def getIndices(x,kernel_size_h,kernel_size_w,stride_h,stride_w):
    
    indices = {}
    stride = 2
    kernel_size = 2
    h_out = (x.size(0) - kernel_size_h) // stride_h + 1
    w_out = (x.size(1) - kernel_size_w) // stride_w + 1

    for i in range(h_out):
        for j in range(w_out):
            start_i = i * stride_h
            start_j = j * stride_w
            end_i = start_i + kernel_size_h
            end_j = start_j + kernel_size_w
            
            indices[(i, j)] = [
                (start_i, start_j),               # top-left
                (start_i, end_j-1),               # top-right
                (end_i-1, start_j),               # bottom-left
                (end_i-1, end_j-1)                # bottom-right
            ]

    return indices


x = torch.arange(0,332*316, dtype=torch.float).reshape(332,316)
#print("Original tensor:\n", x)


padded_x = F.pad(x, (1, 1, 1, 1), 'constant', 0)
#print("\nPadded tensor:\n", padded_x)


# Applying average pooling
y = F.avg_pool2d(padded_x.unsqueeze(0).unsqueeze(0), kernel_size = (2,2), stride=(2,2)).squeeze()
#print("\nAfter avg pooling:\n", y)

y_output  = F.avg_pool2d(y.unsqueeze(0).unsqueeze(0), kernel_size = (2,2), stride=(2,2)).squeeze()
#print("\nAfter avg pooling:\n", y_output)

indices1= getIndices(padded_x,2,2,2,2)


indices2= getIndices(y,2,2,2,2)


print('indices1')
print(len(indices1))
print('indices2')
print(len(indices2))
combined_indices = {}

for key2, value2 in indices2.items():
    temp_dict = {}
    
    # Fetching corresponding regions from indices1 and storing them
    for idx in value2:
        temp_dict[idx] = indices1[idx]
    
    # Storing the indices from indices2
    #temp_dict['indices2'] = value2
    
    combined_indices[key2] = temp_dict

# Now, the combined_indices dict has the values of indices1 stored inside values from indices2


def getIndices_in_orginal_gray(row,col):
    indices_list_gray_image = [index for sublist in combined_indices[(row,col)].values() for index in sublist]
    # print(indices_list_gray_image)

    # print(len(indices_list_gray_image))
    
    return indices_list_gray_image


# print(len(getIndices_in_orginal_gray(2,3)))
# print(getIndices_in_orginal_gray(0,0))




# %%
import pandas as pd
import numpy as np

def getKernel(input):
    
    df = pd.DataFrame(input)

    

    result_df = pd.DataFrame()

    # # Block sizes
    # block_row_size = 83*2
    # block_col_size = 79*2

    # for i in range(0, df.shape[0], block_row_size):
    #     for j in range(0, df.shape[1], block_col_size):
    #         block = df.iloc[i:i+block_row_size, j:j+block_col_size]
    #         #print(block.values)
    #         avg = np.mean(block.values)
    #         std = np.std(block.values)
    #         result_df.loc[i//block_row_size, j//block_col_size] = avg / std


    result_df = np.ones((2,2))
    #print(result_df)
    
    return np.array(result_df)

# %%

  

# %%
import numpy as np


def minusArray(arr2,arr1):
    
    arr1 = arr1.astype(np.int32)
    arr2 = arr2.astype(np.int32)


    arr = arr2 - arr1
    
    return arr
            



def get_diff_gray_image_kernel_list(gray_image_list):

    diff_gray_image_list = []
    diff_gray_image_kernel_list_2_2 = []

    for i in range(len(gray_image_list)-1):
        
        arr = minusArray(gray_image_list[i+1],gray_image_list[i])

        #np.savetxt('test'+str(i)+'.txt',arr , delimiter='\t', fmt='%d')
        
        diff_gray_image_list.append(arr)
        #print(diff_gray_image_list)
    
        
        kernel_arr = getKernel(arr)
        #np.savetxt('test_kernel'+str(i)+'.txt',kernel_arr , delimiter='\t', fmt='%d')
    
        diff_gray_image_kernel_list_2_2.append(kernel_arr)
        

    # print(len(gray_image_list))
    # print(gray_image_list)
    # print(len(diff_gray_image_list))
    # print(diff_gray_image_list)
    # print(len(diff_gray_image_kernel_list_2_2))
    # print(diff_gray_image_kernel_list_2_2)


    return diff_gray_image_list, diff_gray_image_kernel_list_2_2




# %%
# np.savetxt('20220901'+'.txt',gray_image_list[0] , delimiter='\t', fmt='%d')
# np.savetxt('20220902'+'.txt',gray_image_list[1] , delimiter='\t', fmt='%d')

# for i in range(len(diff_gray_image_list)):
#     np.savetxt(str(i+2)+str('-')+str(i+1)+'.txt',diff_gray_image_list[i] , delimiter='\t', fmt='%d')
    

# %%
import torch
import torch.nn as nn

import torch
import torch.nn as nn

def apply_3x3_sharpening(input_tensor,kernel):
    # Assuuming input is a 2D matrix, we reshape it to [1, 1, height, width]
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    # Define the 2x2 sharpening kernel
    kernel_tensor = kernel.unsqueeze(0).unsqueeze(0)
    
    #print(kernel_tensor)
    

    
    kernel_row,kernel_col = len(kernel_tensor), len(kernel_tensor[0])


    # Define the convolutional layer
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_row,kernel_col), stride=(2,2), padding=1, bias=False)
    conv_layer.weight.data = kernel_tensor

    # Apply convolution for sharpening
    output = conv_layer(input_tensor)

    return output[0,0]

def get_kernel_applied_result_in_diff_gray_image(diff_gray_image_list,diff_gray_image_kernel_list_2_2):

    kernel_applied_result_in_diff_gray_image = []
    for i in range(len(diff_gray_image_list)):
        matrix = torch.tensor(diff_gray_image_list[i]).float()
        kernel = torch.tensor(diff_gray_image_kernel_list_2_2[i]).float()

        # print("Input matrix:\n", matrix)

        result = apply_3x3_sharpening(matrix,kernel).detach().numpy()
        # print("\nResult after 2x2 sharpening:\n", result)



        #np.savetxt('test_after_apply_kernel'+str(i)+'.txt',result, delimiter='\t', fmt='%d')

        # print(result.shape)
        
        kernel_applied_result_in_diff_gray_image.append(result)


    return kernel_applied_result_in_diff_gray_image
    
# diff_gray_image_list, diff_gray_image_kernel_list_2_2 = get_diff_gray_image_kernel_list(gray_image_list)
# np.savetxt('kerenel_applied_diff_2-1.txt', get_kernel_applied_result_in_diff_gray_image(diff_gray_image_list,diff_gray_image_kernel_list_2_2)[0], delimiter='\t', fmt='%d')

# %%
import numpy as np

def truncating_array_83_79(original_array):

    # Define the pooling parameters
    pool_size = (2, 2)
    stride = 2

    # Calculate the dimensions of the output array
    output_height = (original_array.shape[1] - pool_size[0]) // stride + 1
    output_width = (original_array.shape[2] - pool_size[1]) // stride + 1


    # Initialize the output array
    output_array = np.zeros((original_array.shape[0], output_height, output_width, 4))

    # Apply 2x2 average pooling with a stride of 2
    for i in range(output_height):
        for j in range(output_width):
            # Define the pooling region
            region = original_array[:, i * stride:i * stride + pool_size[0], j * stride:j * stride + pool_size[1]]
            # Reshape the region to a (4,) array and store it in the output array
            #print(region)
            output_array[:, i, j] = region.reshape(-1, 4)


    data = output_array

    # Reshape the data into a (83, 79) array where each grid has a list of values with shape 4x4858
    reshaped_data = data #data.swapaxes(0, 3).reshape(83, 79, -1)


    return reshaped_data



# %%


# %%
kernel_applied_result_in_diff_gray_image_overall =  []
for year in range(2000,2024):
    filelist = glob.glob(str(year)+'/*.png')
    filelist.sort()

    


    gray_image_list,rgb_image_list = rgb_to_gray_image_conversion(filelist)
    diff_gray_image_list, diff_gray_image_kernel_list_2_2 = get_diff_gray_image_kernel_list(gray_image_list)

  

    temp_arr = get_kernel_applied_result_in_diff_gray_image(diff_gray_image_list,diff_gray_image_kernel_list_2_2)

    print(np.array(temp_arr).shape)
    

    non_zero_values = temp_arr

    kernel_applied_result_in_diff_gray_image_overall += list(non_zero_values)


    






# %%
data_for_calculating_83_79 = truncating_array_83_79(np.array(kernel_applied_result_in_diff_gray_image_overall))



# %%
print(data_for_calculating_83_79.shape)

# %%


import numpy as np
import glob
import datetime
from datetime import date
from datetime import timedelta
# import datetime
# from dateutil.relativedelta import relativedelta
# from datetime import datetime
# import pandas as pd
import os
from netCDF4 import Dataset
import netCDF4 as nc
import xarray as xr
# from osgeo import gdal
# import geopandas as gpd
# from shapely.geometry import MultiPolygon, Polygon, Point
# from scipy import io
import pandas as pd
from scipy.stats import iqr

list_of_files = glob.glob('*lb*q1.txt')  # create the list of file
x= data_for_calculating_83_79

row, col = 83,79
intial = 15
#for kk in np.arange(1.5, , 0.1):
kk = 1.5
print(kk)

lower_bound, q1_for_future_use =np.zeros((row,col)),np.zeros((row,col))

for i in range(row):
    for j in range(col):
        q1=np.percentile(x[:,i,j,:], 25)

        iqr1=iqr(x[:,i,j,:])
        
        lb=q1-kk*iqr1
        lower_bound[i,j]= lb
        q1_for_future_use[i,j] = q1

np.savetxt('lb'+str(intial)+'.txt', lower_bound, delimiter='\t', fmt='%.4f')
np.savetxt('q1'+str(intial)+'.txt', q1_for_future_use, delimiter='\t', fmt='%.4f')

intial += 1

       
        
        
# Create a figure and axis


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 8))

# Display the 2D matrix as an image
im = ax.imshow(q1_for_future_use, cmap='afmhot')  # You can change the colormap to your preference

# Add a colorbar to the plot (optional)
cbar = fig.colorbar(im, ax=ax)

# Set labels and title (optional)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Lower Bound')

# Show the plot
plt.show()   

# %%
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def rgb_to_gray_image_conversion(filelist):

    rgb_image_list = []
    gray_image_list = []

    for file in filelist:
        image_rgb = cv2.imread(file)
        rgb_image_list.append(image_rgb)

        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        gray_image_list.append(gray_image)

    return gray_image_list, rgb_image_list


total_outputs = []


counter1 = 0

counter2 = 0


year_trace = {}
for year in range(2000, 2024):
    
    filelist = glob.glob(str(year)+'/*.png')
    filelist.sort()
    print(filelist)

    gray_image_list, rgb_image_list = rgb_to_gray_image_conversion(filelist)
    


    # year_trace.append(len(gray_image_list)-1)
    for i in range(len(gray_image_list)-1):

        arr = gray_image_list[i+1].astype(np.int32) - gray_image_list[i].astype(np.int32)

        total_outputs.append(arr)

        counter2 +=1
    
    year_trace[year] = [counter1,counter2]

    counter1 = counter2

    

        
    

   

      
print(year_trace)
device = 'cuda'





# Assuming lower_bound and q1_for_future_use are defined elsewhere
lower_bound1 = torch.from_numpy(lower_bound).float().to(device)
q1 = torch.from_numpy(q1_for_future_use).float().to(device)



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1)
        self.mean_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Set kernel weights to [[1, 1], [1, 1]] and prevent them from being updated
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], device=device))
            self.conv1.weight.requires_grad = False
            self.mean_pool.requires_grad = False
            self.max_pool.requires_grad = False


    def forward(self,x):
        org_x = x
        conv_x = self.conv1(x)
        anomaly_discord_array = torch.full_like(x, 1)

        # Remove positive values
        x_negative_removed = torch.where(conv_x > 0, torch.tensor(0.0).to(x.device), conv_x)
        x_abs = torch.abs(x_negative_removed)
        x_max_pooled = self.max_pool(x_abs)
        x_max_pooled_negative = x_max_pooled * -1
        x_mean_pooled = self.mean_pool(conv_x)

        condition1 = x_max_pooled_negative < lower_bound1
        condition2 = (x_mean_pooled / x_max_pooled_negative) > (q1 / lower_bound1)
        condition = condition1 & condition2

        for batch_index in range(condition.shape[0]):
            anomaly_discord_all_row_col_list = []
            get_indices_anomaly = condition[batch_index].nonzero(as_tuple=True)
            indices_anomaly = list(zip(get_indices_anomaly[1].cpu().numpy(), get_indices_anomaly[2].cpu().numpy()))

            for idx in indices_anomaly:
                i, j = idx
                indices_in_org_gray = getIndices_in_orginal_gray(i, j)
                anomaly_discord_all_row_col_list.append(indices_in_org_gray)

            for iterative_list in anomaly_discord_all_row_col_list:
                for item in iterative_list:
                    rowIndex, colIndex = item
                    #if(org_x[batch_index, 0, rowIndex, colIndex]<0):
                    anomaly_discord_array[batch_index, 0, rowIndex, colIndex] = 0 #* org_x[batch_index, 0, rowIndex, colIndex]
        


        return anomaly_discord_array


dataloader = DataLoader(total_outputs, batch_size=1024, shuffle=False)

model = CNNModel().to(device)
model.eval()
# Perform forward pass for each batch
total_images = []

with torch.no_grad():
    for batch in dataloader:
        input = batch.unsqueeze(1)
        input2 = input.float().to(device)
        
        output = model(input2)
        #print(output.shape)
        output = output.squeeze(1)  # Remove channel dimension

        #print(output.shape)

        # Append each item in the output to the total_images list
        for item in output:
            total_images.append(item.cpu().numpy())

            #print(item.cpu().numpy().shape)





start_index = year_trace[2006][0]
train_len = year_trace[2022][1]





total_len= len(total_images)
total_images = np.array(total_images)
print(total_images.shape)

# %%






x_train = total_images[start_index:train_len-1] 
y_train = total_images[start_index+1:train_len] 


_test_len,total_len = year_trace[2000][0], year_trace[2004][1] 

x_test = total_images[_test_len:total_len-1]
y_test = total_images[_test_len+1:total_len]



print(0,train_len)
print(1,train_len+1)
print(_test_len,total_len-1)
print(_test_len+1,total_len)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Checking if CUDA (GPU) is available
device = "cuda"
print(f"Using device: {device}")

# Convert NumPy arrays to PyTorch tensors and move to the GPU
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

x_test_tensor = torch.from_numpy(x_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device)

# Create DataLoader for training and testing
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size  = 32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size =  32, shuffle=False)

print(len(test_loader))

# %%




# %%
print(0,train_len)
print(1,train_len+1)
print(_test_len,total_len-1)
print(_test_len+1,total_len)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # New convolutional layer
        self.relu3 = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)  # New convolutional layer
        x = self.relu3(x)  # New activation function
        
        
        
        return x



model = VAE().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
device = 'cuda'



# Training loop with validation
epochs = 4000
best_val_loss = float('inf')
patience = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for data in train_loader:
        current_day, next_day = data
        current_day, next_day = current_day.unsqueeze(1).float().to(device), next_day.unsqueeze(1).float().to(device)
        
        optimizer.zero_grad()
        recon_next_day = model(current_day)
        loss = criterion(recon_next_day, next_day) #, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in test_loader:
            current_day, next_day = data
            current_day, next_day = current_day.unsqueeze(1).float().to(device), next_day.unsqueeze(1).float().to(device)
            recon_next_day = model(current_day)
            loss = criterion(recon_next_day, next_day) #, mu, logvar)
            val_loss += loss.item()
    
    # train_loss /= len(train_loader.dataset)
    # val_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the best model based on validation loss
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_vae_model_2000_2005_early.pth')
        counter = 0  # Reset patience counter on improvement
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break








# %%


# %%
import matplotlib.pyplot as plt
# Load the best model
path_2016_22 = VAE().to('cpu')
path_2016_22.load_state_dict(torch.load('best_vae_model_2000_2005_early.pth', map_location=torch.device('cpu')))



x_test_tensor = x_test_tensor.cpu()
path_2016_22.eval()
x_test_tensor = x_test_tensor.float()
#Make predictions on the test set using the best model
with torch.no_grad():
    y_pred = path_2016_22(x_test_tensor.unsqueeze(1))  # Add channel dimension

# Reshape the predicted and actual data back to the 2D grid shape
y_pred_2d = y_pred.cpu().numpy().reshape(-1, 332, 316)
y_test_2d = y_test_tensor.cpu().numpy().reshape(-1, 332, 316)

#printing shapes
print('Shape of y_pred_2d:', y_pred_2d.shape)
print('Shape of y_test_2d:', y_test_2d.shape)

# %%


# %%
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Set a threshold value for anomaly detection
threshold = 0.9  # Adjust this value based on your requirements

y_pred_binary, y_test_binary = y_pred_2d, y_test_2d

# Initialize lists to store metrics for each image
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

# Initialize arrays to store aggregated predictions and ground truth labels
y_pred_all = []
y_test_all = []

# Iterate over each image
for i in range(y_pred_binary.shape[0]):
    y_pred_flat = y_pred_binary[i].flatten()
    print(np.min(y_pred_flat),np.max(y_pred_flat))
    y_pred_flat = np.where(y_pred_flat < threshold, 0, 1)
    y_test_flat = y_test_binary[i].flatten()

   
    
    # Append predictions and ground truth labels to the aggregated arrays
    y_pred_all.extend(y_pred_flat)
    y_test_all.extend(y_test_flat)

    # Calculate precision, recall, and F1 score for the current image
    precision = precision_score(y_test_flat, y_pred_flat)
    recall = recall_score(y_test_flat, y_pred_flat)
    f1 = f1_score(y_test_flat, y_pred_flat)
    acc = accuracy_score(y_test_flat, y_pred_flat)

    # Append the scores to the respective lists
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    accuracy_scores.append(acc)

# Calculate confusion matrix




# Print the average scores across all images
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")


# %%
conf_matrix = confusion_matrix(y_test_all, y_pred_all,labels=[1,0])

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)






# %%
print(y_test_all.count(1))
print(y_test_all.count(0))





