#!/bin/env python

from unet import UNet
from argparse import ArgumentParser
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.utils import save_image

parser = ArgumentParser()
parser.add_argument("-ckpt",
	help="what checkpoint to use")
parser.add_argument("-i",
	help="input image")
args = parser.parse_args()

if args.ckpt is None:
	print("Please specify the checkpoint with -ckpt.")
	exit(1)
if args.i is None:
	print("No image input. Use -i")
	exit(1)

un_model = UNet()
un_model.load_state_dict(torch.load(args.ckpt))
un_model.eval()

from PIL import Image
pil_img = Image.open(args.i).convert('RGB')

img = torch.unsqueeze(Compose([ToTensor(), Resize((512,512))])(pil_img),dim=0)
out_img = un_model(img)
msk_img = img*out_img
save_image(out_img[0], "%s-mk.jpg"%args.i[:-4])
save_image(msk_img[0], "%s-fg.jpg"%args.i[:-4])

