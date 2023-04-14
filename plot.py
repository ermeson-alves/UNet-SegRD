import cv2
import utils
from torch.utils.data import DataLoader
from config import *
import numpy as np

train_set = utils.DIARETDBDataset(IMGS_FUNDUS_PATH, MASKS_DIR_PATH, 0)
train_loader = DataLoader(train_set, batch_size = 4, shuffle=True)

batch = next(iter(train_loader))
utils.plot_img_and_mask(np.transpose(batch[0][0], 1,2,0), batch[1][0]) 
# for i, j in zip(batch[0], batch[1]):
#     print(i.shape, j.shape)
#     utils.plot_img_and_mask(np.transpose(i.numpy(), 1,2,0), j.numpy())