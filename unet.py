#!/bin/env python

import torch
from torch import nn
from torchvision.transforms import CenterCrop

# The UNet model
# print("CUDA avaialble: %s"%torch.cuda.is_available())
if torch.cuda.is_available() is not True:
	print("CUDA unavailable. Cannot continue.")
	exit()

class CropAndCopy(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, copy_tensor):
		# print("Copy:\n\t%s\n\t%s"%(x.size(), copied.size()))
		cc = CenterCrop(x.size()[2])(copy_tensor)
		# print("CenterCrop:\n\t"+str(cc.size()))
		return torch.cat((x,cc), dim=1)

class UNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.ModuleList([
			# Going down the U
			nn.Conv2d(3,64,3,padding='same'),
			nn.ReLU(),
			nn.Conv2d(64,64,3, padding='same'),
			nn.ReLU(),

			nn.MaxPool2d(2),

			nn.Conv2d(64, 128, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding='same'),
			nn.ReLU(),

			nn.MaxPool2d(2),

			nn.Conv2d(128, 256, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, padding='same'),
			nn.ReLU(),

			nn.MaxPool2d(2),

			nn.Conv2d(256, 512, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding='same'),
			nn.ReLU(),

			nn.MaxPool2d(2),

			nn.Conv2d(512, 1024, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(1024, 512, 3, padding='same'),
			nn.ReLU(),

			# Reached the bottom and going up		
			nn.Upsample(scale_factor=2),
			CropAndCopy(),
			nn.Conv2d(1024, 512, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(512, 256, 3, padding='same'),
			nn.ReLU(),

			nn.Upsample(scale_factor=2),
			CropAndCopy(),
			nn.Conv2d(512, 256, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(256, 128, 3, padding='same'),
			nn.ReLU(),

			nn.Upsample(scale_factor=2),
			CropAndCopy(),
			nn.Conv2d(256, 128, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(128, 64, 3, padding='same'),
			nn.ReLU(),

			nn.Upsample(scale_factor=2),
			CropAndCopy(),
			nn.Conv2d(128, 64, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding='same'),
			nn.ReLU(),
			nn.Conv2d(64, 1, 1, padding='same'),
			nn.Sigmoid(),

	])

		self.crop_copy_from = [3, 8, 13, 18]
		self.crop_copy_to   = [25, 31, 37, 43]

	def forward(self, x):
		copyidx = 0
		cc = []
		for i in range(len(self.layers)):
			L = self.layers[i]
			if i in self.crop_copy_from:
				cc.append(x)
				x = L(x)
				copyidx += 1
			elif i in self.crop_copy_to:
				copyidx -= 1
				x = L(x, cc[copyidx])
			else:
				copy = ""
				x = L(x)

		return x

	def debug(self):
		x = torch.randn(1, 3, 512, 512)
		copy = ""
		copyidx = 0
		fmt = '(%02d): %-15s%-20s%s'
		cc = []
		print('\n%s: Input = %s'%(self.__class__.__name__,x.size()))
		for i in range(len(self.layers)):
			L = self.layers[i]
			if i in self.crop_copy_from:
				copy = "[ copy_to = L%d ]"%copyidx
				cc.append(x)
				x = L(x)
				copyidx += 1
			elif i in self.crop_copy_to:
				copyidx -= 1
				x = L(x, cc[copyidx])
				copy = "[ copy_from = L%d ]"%copyidx
			else:
				copy = ""
				x = L(x)

			print(fmt%(i,L.__class__.__name__, copy, x.size()))



# Data loader

from torch.utils.data import Dataset
from os.path import join
from os import listdir
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize

class DUTS(Dataset):
	def __init__(self, root=".", testing=False):
		self.test = testing
		if self.test:
			self.image_root  = join(root,"DUTS-TE/DUTS-TE-Image")
			self.target_root = join(root,"DUTS-TE/DUTS-TE-Mask")
		else:
			self.image_root  = join(root,"DUTS-TR/DUTS-TR-Image")
			self.target_root = join(root,"DUTS-TR/DUTS-TR-Mask")
		self.files = listdir(self.image_root)
		self.files.sort()
		self.transform = Compose([ToTensor(), Resize((512, 512))])

	def __getitem__(self, idx):
		file = self.files[idx][:-4]
		image = Image.open(join(self.image_root,"%s.jpg"%file))
		target = Image.open(join(self.target_root,"%s.png"%file))
		return (self.transform(image),
			self.transform(target))

	def __len__(self):
		if self.test:
			return 5019
		else:
			return 10553

	def __repr__(self):
		return 'DUTS Dataset { Train: 10553, Test: 5019 }'



