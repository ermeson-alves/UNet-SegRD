import utils
from torch.utils.data import DataLoader
from utils.config import *
from utils.transformations import *
import albumentations



# training transformations and augmentations
transforms_training = ComposeDouble([
    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01),
])


# # validation transformations
# transforms_validation = ComposeDouble([
#     FunctionWrapperDouble(create_dense_target, input=False, target=True),
#     FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
#     FunctionWrapperDouble(normalize_01)
# ])
# train_set = utils.DIARETDBDataset(IMGS_FUNDUS_PATH, MASKS_DIR_PATH, 0, transform=transformations['train'])
train_set = utils.IDRIDDataset(TRAINSET_IMGS, TRAINSET_DIR_MASKS, 2, transform=transforms_training)

# print de uma amostra
x, y = train_set[0]['image'], train_set[0]['mask']
print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')


train_loader = DataLoader(train_set, batch_size = 4, shuffle=True)

batch = next(iter(train_loader))
print(f"b_imgs shape: {batch['image'].shape} ----- b_masks shape: {batch['mask'].shape}")
utils.plot_img_and_mask(batch['image'], batch['mask'])