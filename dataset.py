from torch.utils.data import Dataset
import torch
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
import monai
from config import * 

def img_loader(image_path: Path,is_mask=False):
    with open(image_path, 'rb') as f:
        if not is_mask:
            img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = torch.as_tensor(img)
            img = img.permute(2,0,1)
            # img = Image.fromarray(img)
            return img
        else:
            # mask = Image.open(image_path).resize((224,224)).convert("L")
            mask = cv2.imread(str(image_path), 0)
            mask = cv2.resize(mask, (224, 224))
            mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)[1]
            # mask = Image.fromarray(mask)
            mask = np.expand_dims(mask, 2)  # (H, W, C) -> C=1
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.permute(2,0,1)

            return mask
            


def create_dir(path:Path):
    if not path.exists():
        if not path.parent.exists():
            create_dir(path.parent)
        path.mkdir()

def adaptar_dataset(root_dir: Path, dir_fundus_imgs: Path, dir_groundtruths_imgs: Path, annotations_path: Path):
    """Com base nos arquivos de anotações .txt do dataset diaretdb1_v1.1, essa função cria uma divisão melhor das
    imagens em TESTSET e TRAINSET para facilitar futuras utilizações desses dados"""

    path_base = Path(root_dir/str(annotations_path.stem).upper())
    create_dir(path_base/dir_fundus_imgs.name)
    labels = pd.read_csv(annotations_path, header=None).sort_values(by=0, ascending=True)
    print("Nova pasta com fundoscopias criada.")
    for dir_masks in ['hardexudates', 'hemorrhages', 'redsmalldots', 'softexudates']:
        create_dir(path_base/'ddb1_groundtruth'/dir_masks)

        for label in labels[0]:
            # Salvar a imagem correspondente das anotações na pasta de fundoscopias:
            img_fundus = img_loader(dir_fundus_imgs/label)
            img_fundus.save(path_base/dir_fundus_imgs.name/label)
            # Salvar a mascara:
            mask = img_loader(dir_groundtruths_imgs/dir_masks/label)
            mask.save(path_base/'ddb1_groundtruth'/dir_masks/label)
        print("Nova pasta com mascaras de lesões criada.")


if not (ROOT_DATASET_PATH):
    url = "https://www.it.lut.fi/project/imageret/diaretdb1/diaretdb1_v_1_1.zip"
    monai.apps.download_and_extract(url, output_dir="./datasets")
    # TESTSET:
    adaptar_dataset(ROOT_DATASET_PATH, IMGS_FUNDUS_PATH, MASKS_DIR_PATH, ANNOTATIONS_TEST_PATH)
    # TRAINSET:
    adaptar_dataset(ROOT_DATASET_PATH, IMGS_FUNDUS_PATH, MASKS_DIR_PATH, ANNOTATIONS_TRAIN_PATH)


class DIARETDBDataset(Dataset):
    def __init__(self, images_dir:Path, masks_dir:Path, class_id=0, transform=None):
        """
        Args:
            image_dir: Path para o diretório de fundoscopias
            mask_dir: Path para o diretório de diretórios de mascaras
            class_id: indice auxiliar para acessar um diretorio de lesões (exsudatos, ma's, hemorragias...) em masks_path. 
        """
        # self.img_labels = pd.read_csv(annotations_file, header=None)
        # self.masks_list_dir = sorted(dir.name for dir in masks_path.iterdir() if dir.is_dir())
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.class_id = class_id
        self.transform = transform
        self.images_paths = []
        self.masks_paths = []
        self.images = []
        self.masks = []
        
        # variavel que armazena qual diretório de lesões será usado
        mask_path4 = sorted(masks_dir.iterdir())[class_id]

        for img_path in images_dir.glob('*.png'):
          mask_path = mask_path4 / img_path.name
          self.images_paths.append(img_path)
          self.masks_paths.append(mask_path)
        #   print(img_path, mask_path)
          self.images.append(img_loader(img_path))
          self.masks.append(img_loader(mask_path, True))


        assert len(self.images) == len(self.masks)
    
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            pass

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        """Retorna a imagem e suas mascaras vazia e cheia, aplica transformações se houver"""
        # array com imagem e mascara
        info = [self.images[idx]]

        info.append(self.masks[idx])
        if self.transform:
            info[0] = self.transform(info[0])
            info[1] = self.transform(info[1])

        # Imagem ndarray
        inputs = np.array(info[0])
        inputs = inputs / 255.
        mask = np.array(info[1]) / 255.
         

        return {'image':inputs, 'mask':mask }