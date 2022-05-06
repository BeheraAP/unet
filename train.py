#!/bin/env python

from unet import UNet,DUTS
from torch.utils.data import DataLoader
import torch
from torch import nn

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("action", help='''whether to 'train' or 'test' the model''')

parser.add_argument("-bs",
	default = 16,
	help="define batch size",
	type=int,
	)
parser.add_argument("-ckpt",
	default = "final.pt",
	help="checkpoint to use for testing",
	)
parser.add_argument("-out",
	default = 'output',
	help="where to save model checkpoint during training",
	)
parser.add_argument("-e",
	default = 10,
	help="define the number of epochs",
	type=int)
args=parser.parse_args()

from torch.optim import Adam
print("CUDA avaialble: %s"%torch.cuda.is_available())
un_model = UNet()
un_model.cuda()
optimizer = Adam(un_model.parameters(), lr=1e-5)
lossfn = nn.BCELoss()

if args.action=='train':
	bs = args.bs
	n_epoch = args.e
	print( '''Training:\n%-20s: %-10s\n%-20s: %-10s\n%-20s: %-10s\n'''
	%("  Epoch", n_epoch, "  Batch size", bs, "  Checkpoint", args.out));
	duts = DataLoader(DUTS(), shuffle=True, batch_size = bs)
	datalen = len(duts)

	for epoch in range(n_epoch):
		ds = 0
		loss  = 0
		for img_cpu,mask_cpu in duts:
			# Move to the GPU
			img = img_cpu.cuda()
			mask = mask_cpu.cuda()

			# model prediction
			mask_pred = un_model(img)
			loss = lossfn(mask_pred,mask)

			# backpropagation
			un_model.zero_grad()
			loss.backward()
			optimizer.step()

			print("Epoch %02d: Dataset: %02.3lf%% Loss: %.5lf"
				%(epoch,ds/105.53,loss),
				end="\r", flush=True,)
			ds+=bs
		if epoch==n_epoch-1:
			PATH = "./%s/final.pt"%(args.out)
		else:
			PATH = "./%s/checkpoint%02d.pt"%(args.out,epoch)
		torch.save(un_model.state_dict(), PATH)
		print("Loss: %.6lf. Saved: %s"%(loss,PATH))

elif args.action=='test':
	bs = args.bs
	duts = DataLoader(DUTS(testing=True), shuffle=False, batch_size=bs)
	datalen = len(duts)
	un_model.load_state_dict(torch.load(args.ckpt))
	un_model.eval()

	loss, ds = 0, 0
	for img_cpu,msk_cpu in duts:

		img = img_cpu.cuda()

		msk_pred = un_model(img)
		msk_real = msk_cpu.cuda()

		loss += lossfn(msk_pred,msk_real).item()

		if 1+ds >= datalen:
			ds = datalen
		else:
			ds += 1

		print("Testing: %.1lf%% Loss: %.5lf"%(ds*100./datalen,loss),
			end="\r", flush=True)
	print("\nThe net testing loss is %.04lf"%(loss/datalen));

else:
	print("Please define whether to 'train' or 'test'. Quitting.")

