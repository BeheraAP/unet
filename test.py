from unet import DUTS
from torch.utils.data import DataLoader

duts = DataLoader(DUTS(), shuffle=True, batch_size=5)
print(duts)

img,msk=iter(duts).next()
print("Img: %s\nMsk:%s"%(img.size(),msk.size()));

masked = img*msk
print("Masked image:",masked.size())

from matplotlib import pyplot

fig = pyplot.figure()
fig.tight_layout()

for i  in range(5):
	fig.add_subplot(5,3,i*3+1)
	pyplot.imshow(img[i].permute(1,2,0))

	fig.add_subplot(5,3,i*3+2)
	pyplot.imshow(msk[i].permute(1,2,0),"gray")

	fig.add_subplot(5,3,i*3+3)
	pyplot.imshow(masked[i].permute(1,2,0))

pyplot.savefig("img.jpg", bbox_inches='tight', dpi=200)
