from dataset import *
from model import Net

import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
from torch.autograd import Variable

from PIL import Image

from torch.autograd import Variable
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('dataroot', help='path to dataset of kaggle ultrasound nerve segmentation')
# parser.add_argument('dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='number of epoch to start')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

args = parser.parse_args()
print(args)

############## dataset processing
dataset = kaggle2016nerve(args.dataroot)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, num_workers=args.workers)

############## create model
model = Net()
if args.cuda:
  model.cuda()

############## resume
if args.resume:
  if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})"
        .format(checkpoint['epoch']))
  else:
    print("=> no checkpoint found at '{}'".format(args.resume))


def save_checkpoint(state, filename='checkpoint.tar'):
  torch.save(state, filename)

############## training
model.train()
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

# print(model.state_dict())

def train(epoch):
  """
  training
  """
  loss_fn = nn.MSELoss()

  for i, (x, y) in enumerate(train_loader):
    x, y_true = Variable(x), Variable(y)
    if args.cuda:
      x.cuda()

    for test in range(1):
      optimizer.zero_grad()
      y_pred = model(x)

      loss = loss_fn(y_pred, y_true)
      loss.backward()

      optimizer.step()

  print('epoch: {}, loss: {}'.format(epoch,loss))

  save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict()
  })

for epoch in range(niter):
  train(epoch)


############ just check test (visualization)
model.eval()
train_loader.batch_size=1

for i, (x,y) in enumerate(train_loader):
  if i >= 1:
    break

  y_pred = model(Variable(x))

y_pred_numpy = y_pred.data.numpy()
pred_img = y_pred_numpy[0,0,:,:]
pred_img = pred_img > 0.5
im = Image.fromarray(np.uint8(pred_img*255))
im.show()
