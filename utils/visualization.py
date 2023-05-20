import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_img_and_mask(batch_imgs: torch.tensor, batch_masks: torch.tensor, normalize_01=False):
    '''Recebe um lote de imagens e um lote de mascaras e plota as imagens e as mascaras respectivas
    Args:
    batch_imgs: shape-> (B,C,H,W)
    batch_masks: shape-> (B,H,W)
    
    '''
    fig, ax = plt.subplots(2, batch_imgs.size()[0], figsize=(4 * batch_imgs.size()[0],7))
    for (i, img), mask in zip(enumerate(batch_imgs), batch_masks):
        if batch_imgs.size()[0]==1:
            if normalize_01:
              img = (img.permute(1,2,0) - img.permute(1,2,0).min())/(img.permute(1,2,0).max() - img.permute(1,2,0).min())
              mask = mask.squeeze()
            else:
              img = img.permute(1,2,0)
              mask = mask.squeeze()  

            ax[0].imshow(img); ax[0].set_xticks(np.arange(0,240,40)); ax[0].set_yticks(np.arange(0,240,40))
            ax[1].imshow(mask, cmap='gray'); ax[1].set_xticks(np.arange(0,240,40)); ax[1].set_yticks(np.arange(0,240,40))
        else:  
            if normalize_01:
              img = (img.permute(1,2,0) - img.permute(1,2,0).min())/(img.permute(1,2,0).max() - img.permute(1,2,0).min())
              mask = (mask.squeeze())
            else:
              img, mask = img.permute(1,2,0), mask.squeeze()
            ax[0,i].imshow(img); ax[0,i].set_xticks(np.arange(0,240,40)); ax[0,i].set_yticks(np.arange(0,240,40))
            ax[1,i].imshow(mask, cmap='gray'); ax[1,i].set_xticks(np.arange(0,240,40)); ax[1,i].set_yticks(np.arange(0,240,40))
              
    fig.suptitle("Imagem de Entrada - Máscara")
                        
    plt.show()