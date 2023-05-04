from pathlib import Path

# Paths Diaretdb1
IMGS_FUNDUS_PATH = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_fundusimages")
MASKS_DIR_PATH = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_groundtruth")
ANNOTATIONS_TRAIN_PATH = Path("datasets/diaretdb1_v_1_1/resources/traindatasets/trainset.txt")
ANNOTATIONS_TEST_PATH = Path("datasets/diaretdb1_v_1_1/resources/testdatasets/testset.txt")


# Path para treino
TRAINSET_IMGS = Path("datasets/diaretdb1_v_1_1/TRAINSET/ddb1_fundusimages")
TRAINSET_DIR_MASKS = Path("datasets/diaretdb1_v_1_1/TRAINSET/ddb1_groundtruth")

# Path para teste
TESTSET_IMGS = Path("datasets/diaretdb1_v_1_1/TESTSET/ddb1_fundusimages")
TESTSET_DIR_MASKS = Path("datasets/diaretdb1_v_1_1/TESTSET/ddb1_groundtruth")


# Lesions:
LESIONS_IDRID = {0: 'MA', 1: 'HE', 2:'EX', 3:'SE', 4:'OD'}


# Hiperparametros:
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
