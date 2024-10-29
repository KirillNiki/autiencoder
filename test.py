import os
import csv
from os.path import join

import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random


DATA_PATH = './dtd/images'
CSV_DATSET_PATH = './datasets_csvs/dataset.csv'
CSV_TRAINDATSET_PATH = './datasets_csvs/train_dataset.csv'
CSV_TESTDATSET_PATH = './datasets_csvs/test_dataset.csv'
MAX_SHAPE = 640
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNConfigInterface():
  def __init__(self, 
               in_channels: int,
               out_channels: int,
               kernel_size: int,
               stride: int = 1,
               padding: int = 0,
               bias: bool = True,
               ):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.bias = bias


class AutoEncoder(torch.nn.Module):
  def __init__(self, 
               encoder_params=[CNNConfigInterface(3, 16, (15, 15))], 
               decoder_params=[CNNConfigInterface(16, 3, (15, 15))],
               ):
    super().__init__()
    self.encoder = torch.nn.ModuleList([
      torch.nn.Conv2d(
        in_channels = configInterface.in_channels,
        out_channels = configInterface.out_channels,
        kernel_size = configInterface.kernel_size,
        stride = configInterface.stride,
        padding = configInterface.padding,
        bias = configInterface.bias,
      ) for configInterface in encoder_params
    ])
    self.maxpol = torch.nn.MaxPool2d(2)
    self.decoder = torch.nn.ModuleList([
      torch.nn.ConvTranspose2d(
        in_channels = configInterface.in_channels,
        out_channels = configInterface.out_channels,
        kernel_size = configInterface.kernel_size,
        stride = configInterface.stride,
        padding = configInterface.padding,
        bias = configInterface.bias,
      ) for configInterface in decoder_params
    ])
    self.upsemlp = torch.nn.Upsample(scale_factor=2)
    
  def forward(self, x):
    for layer in self.encoder:
      x = layer(x)
      x = self.maxpol(x)
      
    for layer in self.decoder:
      x = self.upsemlp(x)
      x = layer(x)
      
    return x


class CustomDataset(Dataset):
  def __init__(self, data_path=CSV_TRAINDATSET_PATH):
    super().__init__()
    self.csv_dataset = pd.read_csv(data_path)
    self.convert_totensor = T.ToTensor()
  
  def __len__(self):
    return len(self.csv_dataset)
  
  def __getitem__(self, idx):
    img_path = self.csv_dataset.iloc[idx, 0]
    with Image.open(img_path) as img:
      tensor = self.convert_totensor(img)
      tensor = tensor / 255
    
    if tensor.shape[-2] < MAX_SHAPE or tensor.shape[-1] < MAX_SHAPE:
      first_dim_pad = (MAX_SHAPE - tensor.shape[-2])
      second_dim_pad = (MAX_SHAPE - tensor.shape[-1])
      tensor = torch.nn.functional.pad(tensor, 
          (second_dim_pad//2, second_dim_pad - second_dim_pad//2, first_dim_pad//2, first_dim_pad - first_dim_pad//2))
    
    return tensor
  

def se_loss(output, target):
  loss = torch.sum((output - target)**2)
  return loss


def generate_csv(test_part=0.2):
  csv_file = open(CSV_DATSET_PATH, 'w', newline='')
  writer = csv.writer(csv_file)
  
  dir_names = os.listdir(DATA_PATH)
  for dir_name in dir_names:
    dir_path = join(DATA_PATH, dir_name)
    
    file_names = os.listdir(dir_path)
    for file_name in file_names:
      if file_name.endswith('.jpg'):
        writer.writerow([join(dir_path, file_name)])
  csv_file.close()
  
  csv_dataset = pd.read_csv(CSV_DATSET_PATH)
  train_csv_file = open(CSV_TRAINDATSET_PATH, 'w', newline='')
  train_writer = csv.writer(train_csv_file)
  test_csv_file = open(CSV_TESTDATSET_PATH, 'w', newline='')
  test_writer = csv.writer(test_csv_file)
  
  test_inds = np.random.choice(range(0, len(csv_dataset)), (int)(len(csv_dataset)*test_part))
  for test_ind in test_inds:
    img_path = csv_dataset.iloc[test_ind, 0]
    test_writer.writerow([img_path])
  test_csv_file.close()

  train_inds = np.delete(np.array(range(0, len(csv_dataset))), test_inds)
  for train_ind in train_inds:
    img_path = csv_dataset.iloc[train_ind, 0]
    train_writer.writerow([img_path])
  train_csv_file.close()


def create_model():
  model = AutoEncoder(
    encoder_params=[
      CNNConfigInterface(3, 32, 7),
      CNNConfigInterface(32, 64, 6),
      CNNConfigInterface(64, 128, 5),
      CNNConfigInterface(128, 256, 3),
    ],
    decoder_params=[
      CNNConfigInterface(256, 128, 3),
      CNNConfigInterface(128, 64, 5),
      CNNConfigInterface(64, 32, 6),
      CNNConfigInterface(32, 3, 7),
    ]
  )
  model.to(device=device)
  optimizer = torch.optim.Adam(model.parameters())
  
  input_size = (3, MAX_SHAPE, MAX_SHAPE)
  summary(model=model, input_size=input_size)
  return model, optimizer


if __name__ == '__main__':
  model, optimizer = create_model()
  train_dataset = CustomDataset(CSV_TRAINDATSET_PATH)
  test_dataset = CustomDataset(CSV_TESTDATSET_PATH)
  
  train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
  
  
  loss_fn = se_loss
  for epoch in range(1000):
    index = 0
    
    for train_data in iter(train_data_loader):
      train_data = train_data.to(device=device)
      prediction = model(train_data)
      loss = loss_fn(prediction, train_data)
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      del train_data
      index += 1
    
    losses = []
    with torch.no_grad():
      for test_data in iter(test_data_loader):
        test_data = test_data.to(device=device)
        prediction = model(test_data)
        loss = loss_fn(prediction, test_data)
        
        losses.append(loss)
        del test_data
    print(sum(losses) / len(losses))
    
torch.save(model, 'model')
