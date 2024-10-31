import torch
from torch import nn


class AutoEncoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = torch.nn.ModuleList([
      nn.Conv2d(3, 64, 3),
      nn.Conv2d(64, 64, 3),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(64, 128, 3),
      nn.Conv2d(128, 128, 3),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(128, 256, 3),
      nn.Conv2d(256, 256, 3),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(256, 512, 3),
      nn.Conv2d(512, 512, 3),
    ])
    self.decoder = torch.nn.ModuleList([
      nn.ConvTranspose2d(512, 128, 3),
      nn.ConvTranspose2d(256, 128, 3),
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.ConvTranspose2d(256, 64, 3),
      nn.ConvTranspose2d(128, 64, 3),
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.ConvTranspose2d(128, 32, 3),
      nn.ConvTranspose2d(64, 32, 5),
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.ConvTranspose2d(64, 16, 7, 2),
      nn.ConvTranspose2d(32, 3, 8, 2),
    ])
    self.relu = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
    
  def forward(self, x):
    skip_connections = []
    for index, layer in enumerate(self.encoder):
      x = layer(x)
      if type(layer) == nn.Conv2d:
        x = self.relu(x)
      
      if (index +1) % 3 != 0:
        skip_connections.append(x)
      
    for index, layer in enumerate(self.decoder):
      if (index +1) % 3 != 0:
        connected = skip_connections.pop(-1)
        x = torch.cat((x, connected), 1)
        
      x = layer(x)
      x = self.relu(x) if index != len(self.decoder) -1 else self.sigmoid(x)
    return x
