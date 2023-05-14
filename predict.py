import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms

from unet import UNet
from utils import plot_img_and_mask


def predict_img(net,
                full_img: np.ndarray,
                device,
                scale_factor=1,
                out_threshold=0.5) -> np.ndarray:
    net.eval()
    img = torch.from_numpy(full_img)
    img = img.permute(2,0,1).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    print(f"Shape tensor de entrada em predict_img(): {img.shape}")

    with torch.no_grad():
        output = net(img).cpu()
        print(f"Shape saida da rede: {output.shape}")
        
        output = F.interpolate(output, (full_img.shape[1], full_img.shape[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    
    print(f"Shape objeto numpy retornado em predict_img(): {mask.squeeze().shape}")
    return mask.long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed') ####
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks') 
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(filename: Path):
        return str(filename.with_suffix('_OUT.png'))

    return args.output or list(map(_generate_name, args.input))

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
        # Carrega a imagem com opencv
        img = cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), (224,224))

        # Retorna a mascara predita como objeto numpy
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            cv2.imwrite(out_filename, mask)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            b_img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
            b_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            plot_img_and_mask(b_img, b_mask)