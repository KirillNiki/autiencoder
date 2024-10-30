import torch


class CNNConfigInterface():
  def __init__(self, 
               out_channels: int,
               kernel_size: int,
               stride: int = 1,
               padding: int = 0,
               bias: bool = True,
              ):
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.bias = bias


class AutoEncoder(torch.nn.Module):
  def __init__(self, 
               params=[CNNConfigInterface(16, 15)],
              ):
    super().__init__()
    self.encoder = torch.nn.ModuleList([
      torch.nn.Conv2d(
        in_channels = 3 if i == 0 else params[i-1].out_channels,
        out_channels = params[i].out_channels,
        kernel_size = params[i].kernel_size,
        stride = params[i].stride,
        padding = params[i].padding,
        bias = params[i].bias,
      ) for i in range(0, len(params), 1)
    ])
    self.decoder = torch.nn.ModuleList([
      torch.nn.ConvTranspose2d(
        in_channels = params[i].out_channels,
        out_channels = 3 if i == 0 else params[i-1].out_channels,
        kernel_size = params[i].kernel_size,
        stride = params[i].stride,
        padding = params[i].padding,
        bias = params[i].bias,
      ) for i in range(len(params)-1, -1, -1)
    ])
    self.maxpol = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.upsemlp = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.relu = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
    
  def forward(self, x):
    for index, layer in enumerate(self.encoder):
      x = layer(x)
      x = self.relu(x)
      
      if index != len(self.encoder) -1:
        x = self.maxpol(x)
      
    for index, layer in enumerate(self.decoder):
      if index != 0:
        x = self.upsemlp(x)
      
      x = layer(x)
      x = self.relu(x) if index != len(self.decoder) -1 else self.sigmoid(x)
    return x
