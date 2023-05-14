import torch
from torch import Tensor
import matplotlib.pyplot as plt

# OBS: true_mask e pred devem estar normalizadas entre 0 e 1
def dice_coeff(true_mask: Tensor, pred: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    '''O epsilon é usado porque pode haver casos em que a mascara é completamente preta e evita divisão por 0'''
    assert true_mask.size() == pred.size()
    assert true_mask.dim() == 3 or not reduce_batch_first

    '''Isso é usado para verificar se o cálculo do coeficiente de Dice será feito para cada máscara 
    individualmente ou para todas as máscaras de um lote'''
    sum_dim = (-1, -2) if true_mask.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (true_mask * pred).sum(dim=sum_dim)
    sets_sum = true_mask.sum(dim=sum_dim) + pred.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum) # Suspeita de que essa linha não é util, já que existe um epsilon

    dice = (inter + epsilon) / (sets_sum + epsilon) 
    return dice.mean()


def multiclass_dice_coeff(true_mask: Tensor, pred: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    # Esse flatten é para transformar em um tensor 2d
    return dice_coeff(true_mask.flatten(0, 1), pred.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(true_mask: Tensor, pred: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    print(f"\nResultado dice_loss: {fn(true_mask, pred, reduce_batch_first=True)}\n")
    return 1 - fn(true_mask, pred, reduce_batch_first=True) # é assim porque grande perda é ruim para o modelo