import torch
import torchvision
from PIL import Image
import os


class save_dataset(torch.utils.data.Dataset):

  def __init__(self, dataset, split='train'):
    self.dataset = dataset
    self.split = split

  def __getitem__(self, i):
      x, y = self.dataset[i]
      path = './DATA/'+self.split+'/'+str(y)+'/'+str(i)+'.png'

      if not os.path.isdir('./DATA/'+self.split+'/'+str(y)):
          os.mkdir('./DATA/'+self.split+'/'+str(y))

      x.save(path)

  def __len__(self):
    return len(self.dataset)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

train_dataset = save_dataset(trainset, split='train')
test_dataset = save_dataset(testset, split='test')

if not os.path.isdir('./DATA'):
    os.mkdir('./DATA')

if not os.path.isdir('./DATA/train'):
    os.mkdir('./DATA/train')

if not os.path.isdir('./DATA/test'):
    os.mkdir('./DATA/test')

for idx, i in enumerate(train_dataset):
    train_dataset[idx]
    print(idx)

for idx, i in enumerate(test_dataset):
    test_dataset[idx]
    print(idx)
