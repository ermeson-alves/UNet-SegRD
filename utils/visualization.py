import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Imagem de Entrada')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mascara (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
import cv2
img1, img2 =cv2.imread("datasets\diaretdb1_v_1_1\TRAINSET\ddb1_fundusimages\image001.png"), cv2.imread("datasets\diaretdb1_v_1_1\TRAINSET\ddb1_groundtruth\hardexudates\image001.png", 0)

plot_img_and_mask(img1, img2)
