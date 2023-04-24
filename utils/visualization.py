import matplotlib.pyplot as plt
import numpy
import torch

def plot_img_and_mask(batch_imgs: torch.tensor, batch_masks: torch.tensor):
    '''Recebe um lote de imagens e um lote de mascaras e plota as imagens e as mascaras respectivas'''
    fig, ax = plt.subplots(batch_imgs.size()[0], 2, figsize=(5.6,12))
    for (i, img), mask in zip(enumerate(batch_imgs), batch_masks):
        ax[i,0].imshow(img.permute(1,2,0))
        ax[i,1].imshow(mask[0], cmap='gray')
        plt.xticks([]), plt.yticks([])

    fig.suptitle("Imagem de Entrada - Máscara")
    # plt.subplots_adjust(bottom=0.1,
    #                     top=0.9)
    plt.show()  
