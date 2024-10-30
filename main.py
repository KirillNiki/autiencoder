import torch
from torch.utils.data import DataLoader
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from environment import *
from model import AutoEncoder, CNNConfigInterface
from dataset import CustomDataset, generate_csv


def se_loss(output, target):
  loss = torch.sum((output - target)**2)
  return loss


def create_model():
  model = AutoEncoder(
    params=[
      CNNConfigInterface(16, 7),
      CNNConfigInterface(32, 6),
      CNNConfigInterface(64, 5),
      CNNConfigInterface(128, 3),
      CNNConfigInterface(256, 3),
    ])
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
  
  loss_fn = torch.nn.MSELoss()
  for epoch in range(1000):
    index = 0
    
    for train_data in iter(train_data_loader):
      train_data = train_data.to(device=device)
      prediction = model(train_data)
      loss = loss_fn(prediction, train_data)
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      index += 1
      del train_data
    
    losses = []
    with torch.no_grad():
      for test_data in iter(test_data_loader):
        test_data = test_data.to(device=device)
        prediction = model(test_data)
        loss = loss_fn(prediction, test_data)
        
        losses.append(loss)
        del test_data
    print(epoch, (sum(losses) / len(losses)).item())
    
torch.save(model, 'model')
