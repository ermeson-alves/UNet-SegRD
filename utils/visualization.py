import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_img_and_mask(batch_imgs: torch.tensor, batch_masks: torch.tensor):
    '''Recebe um lote de imagens e um lote de mascaras e plota as imagens e as mascaras respectivas'''
    fig, ax = plt.subplots(batch_imgs.size()[0], 2, figsize=(5,15))
    for (i, img), mask in zip(enumerate(batch_imgs), batch_masks):
        # Obs: o tipo teve que ser mudado para torch.long para a imagem de entrada
        ax[i,0].imshow(img.permute(1,2,0)); ax[i, 0].set_xticks(np.arange(0,240,40)); ax[i, 0].set_yticks(np.arange(0,240,40))
        ax[i,1].imshow(mask.squeeze(), cmap='gray'); ax[i, 1].set_xticks(np.arange(0,240,40)); ax[i, 1].set_yticks(np.arange(0,240,40))
        
    fig.suptitle("Imagem de Entrada - Máscara")
    plt.subplots_adjust(left=0.095,
                        bottom=0.048,
                        right=0.988,
                        top=0.94,
                        wspace=0.336,
                        hspace=0.22)
                        
    plt.show()  

