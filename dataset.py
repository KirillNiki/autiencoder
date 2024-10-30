import os
from os.path import join
import csv

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image

from environment import *


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
