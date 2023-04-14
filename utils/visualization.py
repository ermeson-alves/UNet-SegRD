import matplotlib.pyplot as plt
import numpy


def plot_img_and_mask(img:numpy.ndarray, mask: numpy.ndarray ):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Imagem de Entrada')
    ax[0].imshow(img)

    ax[1].set_title(f'Mascara')
    ax[1].imshow(mask, cmap='gray')

    plt.xticks([]), plt.yticks([])
    plt.show()


