from pathlib import Path

# Paths necessarios (Para DIARETDB1):
ROOT_DATASET_PATH = Path("datasets/diaretdb1_v_1_1")
IMGS_FUNDUS_PATH = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_fundusimages")
MASKS_DIR_PATH = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_groundtruth")
ANNOTATIONS_TRAIN_PATH = Path("datasets/diaretdb1_v_1_1/resources/traindatasets/trainset.txt")
ANNOTATIONS_TEST_PATH = Path("datasets/diaretdb1_v_1_1/resources/testdatasets/testset.txt")

# Path para as imagens de fundoscopia de treino e para o diretorio de mascaras de treino
TRAINSET_IMGS = Path("datasets/diaretdb1_v_1_1/TRAINSET/ddb1_fundusimages")
TESTSET_IMGS = Path("datasets/diaretdb1_v_1_1/TESTSET/ddb1_fundusimages")

# Path para as imagens de fundoscopia de teste e para o diretorio de mascaras de teste
TRAINSET_DIR_MASKS = Path("datasets/diaretdb1_v_1_1/TRAINSET/ddb1_groundtruth")
TESTSET_DIR_MASKS = Path("datasets/diaretdb1_v_1_1/TESTSET/ddb1_groundtruth")