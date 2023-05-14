import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_img_and_mask(batch_imgs: torch.tensor, batch_masks: torch.tensor):
    '''Recebe um lote de imagens e um lote de mascaras e plota as imagens e as mascaras respectivas'''
    fig, ax = plt.subplots(2, batch_imgs.size()[0], figsize=(4 * batch_imgs.size()[0],7))
    for (i, img), mask in zip(enumerate(batch_imgs), batch_masks):
        # Obs: o tipo teve que ser mudado para torch.long para a imagem de entrada
        if batch_imgs.size()[0]==1:
            ax[0].imshow(img.permute(1,2,0)); ax[0].set_xticks(np.arange(0,240,40)); ax[0].set_yticks(np.arange(0,240,40))
            ax[1].imshow(mask.squeeze(), cmap='gray'); ax[1].set_xticks(np.arange(0,240,40)); ax[1].set_yticks(np.arange(0,240,40))
        else:  
            ax[0,i].imshow(img.permute(1,2,0)); ax[0,i].set_xticks(np.arange(0,240,40)); ax[0,i].set_yticks(np.arange(0,240,40))
            ax[1,i].imshow(mask.squeeze(), cmap='gray'); ax[1,i].set_xticks(np.arange(0,240,40)); ax[1,i].set_yticks(np.arange(0,240,40))
            

    fig.suptitle("Imagem de Entrada - Máscara")
    # plt.subplots_adjust(left=0.095,
    #                     bottom=0.048,
    #                     right=0.988,
    #                     top=0.94,
    #                     wspace=0.336,
    #                     hspace=0.22)
                        
    plt.show()  


#######################################################################################################################################################################

