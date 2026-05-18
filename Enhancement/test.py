# # Learning Enriched Features for Fast Image Restoration and Enhancement
# # Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
# # IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
# # https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.UHDM_arch import UHDM
from skimage import img_as_ubyte

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def self_ensemble(x, s, model):
    def forward_transformed(x, s, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
            s = torch.flip(s, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
            s = torch.flip(s, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
            s = torch.rot90(s, dims=(-2, -1))
        x = model(x, s)[0]
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, s, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)


parser = argparse.ArgumentParser(description='Image Enhancement using MIRNet-v2')

parser.add_argument('--input_dir', default='/home/min/Documents/ntire25/data/test/Test_Input', type=str, help='Directory of validation images')
parser.add_argument('--input_s_dir', default='test/input_s', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='test-results-ntire25', type=str, help='Directory for results')
parser.add_argument('--weights', default='weights/model.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='25train', type=str, help='Test Dataset')

args = parser.parse_args()


####### Load yaml #######
yaml_file = 'Options/Ntire25_LowLight.yml'
weights = args.weights

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = UHDM(**x['network_g'])

total = sum([param.nelement() for param in model_restoration.parameters()])
print('total parameters:', total)

checkpoint = torch.load(weights)
checkpoint_name = os.path.basename(weights).split('.')[0]
if checkpoint_name.split('_')[-1] == 'complete':
    model_restoration.load_state_dict(checkpoint, strict=True)
else:
    model_restoration.load_state_dict(checkpoint['params'], strict=False)
print("===>Testing using weights: ",weights)
model_restoration.cuda()
model_restoration.eval()


factor = 32
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, checkpoint_name)
os.makedirs(result_dir, exist_ok=True)


input_paths = natsorted(glob(os.path.join(args.input_dir, '*.png')) + glob(os.path.join(args.input_dir, '*.JPG')))
input_s_paths = natsorted(glob(os.path.join(args.input_s_dir, '*.png')) + glob(os.path.join(args.input_s_dir, '*.JPG')))

psnr = []
with torch.inference_mode():
    for inp_path, input_s_path in tqdm(zip(input_paths,input_s_paths), total=len(input_s_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path))/255.
        img_s = np.float32(utils.load_img(input_s_path))/255.

        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        img_s = torch.from_numpy(img_s).permute(2,0,1)
        input_s_= img_s.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 32
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        input_s_ = F.pad(input_s_, (0,padw,0,padh), 'reflect')
        
        restored = self_ensemble(input_, input_s_, model_restoration)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        
        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0]+'.png')), img_as_ubyte(restored))
