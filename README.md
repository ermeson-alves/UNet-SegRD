# UNet-SegRD

Baseado em: [Medical image segmentation with TorchIO, MONAI & PyTorch Lightning](https://github.com/Project-MONAI/tutorials/blob/main/modules/TorchIO_MONAI_PyTorch_Lightning.ipynb) e [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)


Estrutura de um projeto Pytorch para ```segmentação semântica:```

project/<br>
* data/<br>
-- train/<br>
---- imgs/<br>
-------- image_1.png<br>
-------- image_2.png<br>
---- masks/<br>
-------- mask_1.png<br>
-------- mask_2.png<br>
-- val/<br>
---- imgs/<br>
-------- image_1.png<br>
-------- image_2.png<br>
---- masks/<br>
-------- mask_1.png<br>
-------- mask_2.png<br>
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
