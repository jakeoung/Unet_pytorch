import torch.utils.data as data
import torch

from scipy.ndimage import imread
import os
import os.path
import glob

import numpy as np

from torchvision import transforms

def make_dataset(root, train=True):

  dataset = []

  if train:
    dir = os.path.join(root, 'train')

    for fGT in glob.glob(os.path.join(dir, '*_mask.tif')):
      fName = os.path.basename(fGT)
      fImg = fName[:-9] + '.tif'

      dataset.append( [os.path.join(dir, fImg), os.path.join(dir, fName)] )

  return dataset

class kaggle2016nerve(data.Dataset):
  """
  Read dataset of kaggle ultrasound nerve segmentation dataset
  https://www.kaggle.com/c/ultrasound-nerve-segmentation
  """

  def __init__(self, root, transform=None, train=True):
    self.train = train

    # we cropped the image
    self.nRow = 400
    self.nCol = 560

    if self.train:
      self.train_set_path = make_dataset(root, train)

  def __getitem__(self, idx):
    if self.train:
      img_path, gt_path = self.train_set_path[idx]

      img = imread(img_path)
      img = img[0:self.nRow, 0:self.nCol]
      img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
      img = (img - img.min()) / (img.max() - img.min())
      img = torch.from_numpy(img).float()

      gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
      gt = np.atleast_3d(gt).transpose(2, 0, 1)
      gt = gt / 255.0
      gt = torch.from_numpy(gt).float()

      return img, gt

  def __len__(self):
    if self.train:
      return 5635
    else:
      return 5508