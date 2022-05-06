# UNet
This is my own implementation of the UNet for image segmentation. The original
paper can be found [here](http://arxiv.org/abs/1505.04597).

## Requirements
The following libraries is required to train the network and do prediction.

- [Torch](https://pytorch.org)
- [TorchVision](https://pytorch.org)
- [Pillow](https://pillow.readthedocs.io/en/stable/index.html)

I have tried to keep the dependencies to a minimum. I believe that there is no
version specific code, and hence it _should_ run on any reasonably newish version
of Torch. Please let me know it if doesn't.

I have tested this implementation on the following setup:

- Torch 1.11.0
- PIL 8.4.0
- CUDA 11
- Void Linux

## Dataset
For training and testing, the following DUTS dataset was used, which can be found
[here](http://saliencydetection.net/duts/). Download both `DUTS-TE.zip` and
`DUTS-TR.zip` and extract it. The extracted directory names should be,

- `DUTS-TE`
- `DUTS-TR`

and present in the project root. If you extract it in sub-directory or anywhere
else, you have to use the `-d` flag in `train.py`.

#### Dataset error
I faced some problem with the dataset. The mask should be a grayscale image, i.e,
of shape `[1,H,W]` but for images are in sRGB colorspace, `[3,H,W]`. You can identify
those images by issuing,

```bash
identify * | grep -vi Gray | awk '{print $1}'
```

`identify` is a tool that comes with the [ImageMagick](https://imagemagick.org/)
package. And ImageMagick convert these images to Gray colorspace too:

```bash
for i in $(identify * | grep -vi Gray | awk '{print $1}'); do
    convert $i -colorspace Gray $i;
done
```
Run this command in the mask directory: `DUTS-TE-Mask` and `DUTS-TR-Mask`.

## Running UNet
### Setup
The following steps should setup the inplementation to train, test and inference.

- Clone this repo

```bash
git clone 'https://github.com/annadabehera/unet.git'
cd unet
```

- Download the dataset from [here](http://saliencydetection.net/duts/) and extract it.
Use either `7z` or `unzip` for extracting the archive.

```bash
unzip DUTS-TR.zip DUT-TE.zip
7z x DUTS-TR.zip DUT-TE.zip
```

- Setup the environment. You may use your preferred virtual environment method.
I use [mamba](https://mamba.readthedocs.io/en/latest/index.html) to manage it.

```bash
mamba create --name unet
conda activate unet
```

- Install the required packages.

```bash
mamba install -c pytorch torch torchvision
mamba install pillow
```

### Training and testing 
The `train.py` contains the code to both train and test the network.

```bash
$ python train.py --help
usage: train.py [-h] [-bs BS] [-ckpt CKPT] [-out OUT] [-e E] action

positional arguments:
  action      whether to 'train' or 'test' the model

optional arguments:
  -h, --help  show this help message and exit
  -bs BS      define batch size
  -ckpt CKPT  checkpoint to use for testing
  -out OUT    where to save model checkpoint during training
  -e E        define the number of epochs
```

To train the model and save the checkpoints in `./output` directory,

```bash
mkdir output
python train.py train -e 200 -bs 32 -out ./output
```

To test the model with the trained weights stored in `./output` directory,

```bash
python train.py test -ckpt ./output/final.pt -bs 32
```

If you see, `RuntimeError: CUDA out of memory.` error, reduce the batch size with
the `-bs` flag.

### Inference
The `infer.py` file will take input an image and output both the mask and masked
image.

```bash
$ python infer.py --help
usage: infer.py [-h] [-ckpt CKPT] [-i I]

optional arguments:
  -h, --help  show this help message and exit
  -ckpt CKPT  what checkpoint to use
  -i I        input image
```

For a image `example.jpg`, issue the command,

```bash
$ python infer.py -ckpt ./output/final.pt -i example.jpg
```

and this should output two images, `example-mk.jpg` and `example-fg.jpg` for the
mask and masked foreground image in the same directory.

## Known issues
The following issues are known and is being worked upon,

- The output image is of size `[C, 512, 512]` no matter the input size. The channel
`C`, is 1 for mask, `*-mk.jpg` and 3 for the backgound removed image, `*-fg.jpg`.

## License
The work is released in public domain. Feel free to use it with or without credit.
I take no responsibility, whatsoever, for what you do with this code.


