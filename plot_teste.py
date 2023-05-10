import utils
from utils.dataset import *
from torch.utils.data import DataLoader
from utils.config import *
import numpy as np
import torchvision.transforms as transforms

transformations = {
    'test': transforms.Compose([
                        # transforms.ToPILImage(),
                        transforms.Resize((224,224)),
#                         transforms.Grayscale(1),
                        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        # transforms.ToTensor()]),
                        ]),
     'train': transforms.Compose([
                        #    transforms.ToPILImage(),
                           transforms.Resize(size=(224,224)),
                        #    transforms.RandomRotation(degrees = 15), 
                        #    transforms.RandomHorizontalFlip(p = 0.005),
                        #    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        #    transforms.Grayscale(1),
                        #    transforms.ToTensor()
                           ])
}

# train_set = utils.DIARETDBDataset(IMGS_FUNDUS_PATH, MASKS_DIR_PATH, 0, transform=transformations['train'])
train_set = utils.IDRIDDataset(TRAINSET_IMGS,
                               TRAINSET_DIR_MASKS,
                               2,True)

# print de uma amostra
x, y = train_set[0]['image'], train_set[0]['mask']
print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')


train_loader = DataLoader(train_set, batch_size = 4, shuffle=True)

batch = next(iter(train_loader))
print(f"b_imgs shape: {batch['image'].shape} ----- b_masks shape: {batch['mask'].shape}")
utils.plot_img_and_mask(batch['image'], batch['mask'])