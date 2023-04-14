from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
import monai
from config import * 

def pil_loader(image_path,is_mask=False):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        h, w = img.size
        if not is_mask:
            return img.resize((h//2, w//2)).convert('RGB')
            # return img.convert('RGB')
        else:
            return img.resize((h//2, w//2)).convert('L')
            # return img.convert('L')


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
            img_fundus = pil_loader(dir_fundus_imgs/label)
            img_fundus.save(path_base/dir_fundus_imgs.name/label)
            # Salvar a mascara:
            mask = pil_loader(dir_groundtruths_imgs/dir_masks/label)
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
          self.images.append(pil_loader(img_path))
          self.masks.append(pil_loader(mask_path, True))


        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        """Retorna a imagem e suas mascaras vazia e cheia, aplica transformações se houver"""
        # array com imagem e mascara
        info = [self.images[idx]]

        info.append(self.masks[idx])
        if self.transform:
          info = self.transform(info)

        # Imagem ndarray
        inputs = np.array(info[0])

        if inputs.shape[2] == 3:
          # transpoem as imagens e normaliza os pixels
          inputs = np.transpose(np.array(info[0]), (2,0,1))
          inputs = inputs / 255.
        
        if len(info)>1:
          mask = np.array(info[1]) / 255.
         

          return inputs, mask


    


