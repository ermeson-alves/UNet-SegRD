import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.dataset import DIARETDBDataset
from unet import UNet
from utils import plot_img_and_mask
import matplotlib.pyplot as plt

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(np.asarray(full_img.convert("RGB")).copy())
    img = img.unsqueeze(0).permute(0,3,1,2)
    img = img.to(device=device, dtype=torch.float32)
    print(f"Shape tensor de entrada em predict_img(): {img.shape}")

    with torch.no_grad():
        output = net(img).cpu()
        print(f"Shape saida da rede: {output.shape}")
        
        # plt.imshow(output.squeeze().permute(1,2,0)[:, :, 1]); plt.show()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
            print(f"Shape mask: {mask.shape}")
            plt.imshow(mask.squeeze(), cmap='gray'); plt.show()
        else:
            mask = torch.sigmoid(output) > out_threshold
    
    print(f"Shape objeto numpy retornado em predict_img(): {mask[0].long().squeeze().numpy().shape}")
    return mask[0].long().squeeze().numpy() 


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

# Retorna a imagem como PIL Image
def mask_to_image(mask: np.ndarray):
    mask = np.uint8(mask)
    return Image.fromarray(mask)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)

    # Carregando os checkpoints do modelo
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    # Para cada imagem de entrada...
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        # Carrega a imagem com pillow
        img = Image.open(filename)

        # Retorna a mascara predita como objeto numpy
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            print(f"out_filename: {out_filename}")
            result = mask_to_image(mask)
            print(f"Shape result: {result.size}")
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            print(type(img), type(mask))
            plot_img_and_mask(img, mask)