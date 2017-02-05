from dataset import *
from model import Net

import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import torch.backends.cudnn as cudnn
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
parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')
parser.add_argument('--output_name', default='checkpoint___.tar', type=str, help='output checkpoint filename')

args = parser.parse_args()
print(args)

############## dataset processing
dataset = kaggle2016nerve(args.dataroot)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                           num_workers=args.workers, shuffle=True)

############## create model
model = Net(args.useBN)
if args.cuda:
  model.cuda()
  cudnn.benchmark = True

############## resume
if args.resume:
  if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))

    if args.cuda == False:
      checkpoint = torch.load(args.resume, map_location={'cuda:0':'cpu'})

    args.start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, loss {})"
        .format(checkpoint['epoch'], checkpoint['loss']) )
  else:
    print("=> no checkpoint found at '{}'".format(args.resume))


def save_checkpoint(state, filename=args.output_name):
  torch.save(state, filename)

############## training
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
model.train()

def train(epoch):
  """
  training
  """
  loss_fn = nn.MSELoss()
  if args.cuda:
    loss_fn = loss_fn.cuda()

  loss_sum = 0

  for i, (x, y) in enumerate(train_loader):
    x, y_true = Variable(x), Variable(y)
    if args.cuda:
      x = x.cuda()
      y_true = y_true.cuda()

    for ii in range(1):
      y_pred = model(x)

      loss = loss_fn(y_pred, y_true)
      
      optimizer.zero_grad()
      loss.backward()
      loss_sum += loss.data[0]

      optimizer.step()
 
    if i % 5 == 0:
      print('batch no.: {}, loss: {}'.format(i, loss.data[0]))

  print('epoch: {}, epoch loss: {}'.format(epoch,loss.data[0]/len(train_loader) ))

  save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'loss': loss.data[0]/len(train_loader)
  })

for epoch in range(args.niter):
  train(epoch)


############ just check test (visualization)

def showImg(img, binary=True, fName=''):
  """
  show image from given numpy image
  """
  img = img[0,0,:,:]

  if binary:
    img = img > 0.5

  img = Image.fromarray(np.uint8(img*255), mode='L')

  if fName:
    img.save('assets/'+fName+'.png')
  else:
    img.show()


model.eval()
train_loader.batch_size=1

for i, (x,y) in enumerate(train_loader):
  if i >= 11:
    break

  y_pred = model(Variable(x))
  showImg(x.numpy(), binary=False, fName='ori_'+str(i))
  showImg(y_pred.data.numpy(), binary=False, fName='pred_'+str(i))
  showImg(y.numpy(), fName='gt_'+str(i))

