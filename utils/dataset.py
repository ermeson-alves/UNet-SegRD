from torch.utils.data import Dataset
import torch
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
import monai
from .config import * 

def img_loader(image_path: Path,is_mask=False, dataset='IDRID'):
    with open(image_path, 'rb') as f:
        if dataset=="DIARETDB":
            if not is_mask:
                img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                # img = torch.as_tensor(img)
                # img = img.permute(2,0,1)
                return img
            else:
                mask = cv2.imread(str(image_path), 0)
                mask = cv2.resize(mask, (224, 224))
                mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)[1]
                # mask = np.expand_dims(mask, 2)  # (H, W, C) -> C=1
                # mask = torch.as_tensor(mask, dtype=torch.uint8)
                # mask = mask.permute(2,0,1)
        elif dataset=="IDRID":
            if not is_mask:
                img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                # img = torch.as_tensor(img)
                # img = img.permute(2,0,1)
                # img = Image.fromarray(img)
                return img
            else:
                mask = cv2.imread(str(image_path))[:, :, 2]
                mask = cv2.resize(mask, (224, 224))
                # mask = np.expand_dims(mask, 2)  # (H, W, C) -> C=1
                # mask = torch.as_tensor(mask, dtype=torch.uint8)
                # mask = mask.permute(2,0,1)    

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


if not (Path("datasets/diaretdb1_v_1_1")):
    url = "https://www.it.lut.fi/project/imageret/diaretdb1/diaretdb1_v_1_1.zip"
    monai.apps.download_and_extract(url, output_dir="./datasets")
    # TESTSET:
    adaptar_dataset(Path("datasets/diaretdb1_v_1_1"), IMGS_FUNDUS_PATH, MASKS_DIR_PATH, ANNOTATIONS_TEST_PATH)
    # TRAINSET:
    adaptar_dataset(Path("datasets/diaretdb1_v_1_1"), IMGS_FUNDUS_PATH, MASKS_DIR_PATH, ANNOTATIONS_TRAIN_PATH)


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

        for img_path in sorted(images_dir.glob('*.png')):
          mask_path = mask_path4 / img_path.name
          self.images_paths.append(img_path)
          self.masks_paths.append(mask_path)
          self.images.append(img_loader(img_path,dataset='DIARETDB'))
          self.masks.append(img_loader(mask_path,is_mask=True, dataset='DIARETDB'))


        assert len(self.images) == len(self.masks)
    
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        """Retorna a imagem e sua mascara vazia, aplica transformações se houver"""
        input = self.images[idx]
        true_mask = self.masks[idx]

        if self.transform: # conferir se a transformação deve ser aplicada assim mesmo!
            input, true_mask = self.transform(input, true_mask)

        # Imagem ndarray
        input = torch.from_numpy(input).type(torch.float32)
        true_mask = torch.from_numpy(true_mask).type(torch.long)
         

        return {'image':input, 'mask':true_mask }
    


class IDRIDDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, class_id=0, transform=None):
        super().__init__()
        """
        Args:
            image_dir: Path para o diretório de fundoscopias
            mask_dir: Path para o diretório de diretórios de mascaras
            class_id: A ordem muda aqui! 
        """
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

        for img_path in sorted(images_dir.glob('*.jpg')):
          mask_path = mask_path4 / (img_path.stem+'_'+LESIONS_IDRID[class_id]+'.tif')
          self.images_paths.append(img_path)
          self.masks_paths.append(mask_path)
        #   print(img_path, mask_path)
          self.images.append(img_loader(img_path,dataset='IDRID'))
          self.masks.append(img_loader(mask_path,is_mask=True, dataset='IDRID'))

    def __len__(self):
        return len(self.images_paths) 

    def __getitem__(self, idx):
        """Retorna a imagem e sua mascara vazia em forma de tensor, aplica transformações se houver"""
        input = self.images[idx]
        true_mask = self.masks[idx]

        if self.transform: # conferir se a transformação deve ser aplicada assim mesmo!
            input, true_mask = self.transform(input, true_mask)

        # Imagem ndarray
        input = torch.from_numpy(input).type(torch.float32).permute(2,0,1)
        true_mask = torch.from_numpy(true_mask).type(torch.long)
         

        return {'image':input, 'mask':true_mask }       
        