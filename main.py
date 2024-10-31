import sys
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from environment import *
from model import AutoEncoder
from dataset import CustomDataset, generate_csv


def se_loss(output, target):
  loss = torch.sum((output - target)**2)
  return loss


def create_model():
  model = AutoEncoder()
  model.to(device=device)
  optimizer = torch.optim.Adadelta(params=model.parameters(), lr=LEARNING_RATE)
  
  input_size = (3, MAX_SHAPE, MAX_SHAPE)
  summary(model=model, input_size=input_size)
  return model, optimizer


if __name__ == '__main__':
  model, optimizer = create_model()
  train_dataset = CustomDataset(CSV_TRAINDATSET_PATH)
  test_dataset = CustomDataset(CSV_TESTDATSET_PATH)
  
  train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
  
  
  test = next(iter(train_data_loader))
  
  loss_fn = torch.nn.MSELoss()
  for epoch in range(EPOCHS):
    index = 0
    losses = []
    
    for train_data in [test]:
      train_data = train_data.to(device=device)
      optimizer.zero_grad()
      
      prediction = model(train_data)
      loss = loss_fn(prediction, train_data)
      loss.backward()
      optimizer.step()
      
      losses.append(loss)
      index += 1
    
    # losses = []
    # with torch.no_grad():
    #   for test_data in [test]:
    #     test_data = test_data.to(device=device)
    #     prediction = model(test_data)
    #     loss = loss_fn(prediction, test_data)
    
    #     losses.append(loss)
    #     del test_data
    print(epoch, (sum(losses) / len(losses)).item())
    sys.stdout.flush()
    
  torch.save(model, 'model')



  transform = T.ToPILImage()
  test = test.to(device=device)
  test = test[0][None, :]
  result = model(test)
  
  test = test.to(device='cpu')
  image0 = transform(test[0] * 255)
  image0.save('./test/image0.jpg')
  
  result = result.to(device='cpu')
  image1 = transform(result[0] * 255)
  image1.save('./test/image1.jpg')