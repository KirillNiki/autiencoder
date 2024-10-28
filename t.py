import numpy as np
from PIL import Image
import torch
import math

import torch.nn.functional as F


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict 


# test = unpickle('cifar-10-batches-py/data_batch_1')
# test = test[b'data'][1000, :]

# test = torch.tensor(np.concatenate((
#     np.reshape(test[:1024], (32, 32, 1)),
#     np.reshape(test[1024:2048], (32, 32, 1)),
#     np.reshape(test[2048:30072], (32, 32, 1))),
# axis=-1))

betas = torch.linspace(0.0001, 0.002, 200)
# betas = torch.cos(betas * (2 * math.pi))
alphas = 1. - betas

alphas_cumprod = torch.cumprod(alphas, axis=0)


sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


with Image.open('./image0.jpg') as im:
    t = torch.tensor(np.array(im))
    test = t / 255


i = 50

noise = torch.randn_like(test)
test = sqrt_alphas_cumprod[i] * test + sqrt_one_minus_alphas_cumprod[i] * noise

image = Image.fromarray((np.int8)(test.numpy() * 255), "RGB")
image.save(f'image.png')
