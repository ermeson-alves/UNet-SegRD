# UNet-SegRD

Baseado em: [Medical image segmentation with TorchIO, MONAI & PyTorch Lightning](https://github.com/Project-MONAI/tutorials/blob/main/modules/TorchIO_MONAI_PyTorch_Lightning.ipynb) e [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

* **checkpoints/** é o diretório destinado para os estados pós treinamento.
* **datasets/** é onde a parte destinada a segmentação do IDRiD dataset deve ficar, também destinada a outros datasets. 
* **unet/** contém os códigos python de implementação do modelo com base na biblioteca PyTorhc. 
* **utils/** é um módulo que possui o arquivo de configurações básicas, um arquivo de transformações e um arquivo útil de visualização.

O arquivo principal de treinamento é o train.py na raiz do projeto e possui parâmetros de linha de comando definidos conforme [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet):

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
 ```
<br><br>Estrutura de um projeto Pytorch para ```segmentação semântica:```

project/<br>
* data/<br>
-- train/<br>
---- imgs/<br>
---- masks/<br>
-- val/<br>
---- imgs/<br>
---- masks/<br>
* models/<br>
-- unet.py<br>
-- ...<br>
* utils/<br>
-- dataset.py<br>
-- transforms.py<br>
-- ...<br>
train.py<br>
eval.py<br>
config.yaml<br>
requirements.txt<br><br>

### Pendencias (04/05/2023): 
    Treinamento
    Implementar pré-processamentos/ Enhancements
    Validação
    Otimizar hiperparametros
    Teste

Obs: Em fase de adaptação de código de dataset
