from pathlib import Path

# Paths Diaretdb1
# IMGS_FUNDUS_PATH = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_fundusimages")
# MASKS_DIR_PATH = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_groundtruth")
# ANNOTATIONS_TRAIN_PATH = Path("datasets/diaretdb1_v_1_1/resources/traindatasets/trainset.txt")
# ANNOTATIONS_TEST_PATH = Path("datasets/diaretdb1_v_1_1/resources/testdatasets/testset.txt")

# #--->> Path para treino
# TRAINSET_IMGS = Path("datasets/diaretdb1_v_1_1/TRAINSET/ddb1_fundusimages")
# TRAINSET_DIR_MASKS = Path("datasets/diaretdb1_v_1_1/TRAINSET/ddb1_groundtruth")
# #--->> Path para teste
# TESTSET_IMGS = Path("datasets/diaretdb1_v_1_1/TESTSET/ddb1_fundusimages")
# TESTSET_DIR_MASKS = Path("datasets/diaretdb1_v_1_1/TESTSET/ddb1_groundtruth")


# Paths IDRID
#--->> Path para treino
TRAINSET_IMGS = Path("datasets/A. Segmentation/1. Original Images/a. Training Set")
TRAINSET_DIR_MASKS = Path("datasets/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set")
#--->> Path para teste
TESTSET_IMGS = Path("datasets/A. Segmentation/1. Original Images/b. Testing Set")
TESTSET_DIR_MASKS = Path("datasets/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set")

# Lesions:
LESIONS_IDRID = {0: 'MA', 1: 'HE', 2:'EX', 3:'SE', 4:'OD'}
MEAN_STD_LESIONS_TRAIN = {'MA':([116.2852,  56.3344,  16.0175], [82.1312, 42.6342, 21.8126]), 
                          'HE':([118.1395,  56.3422,  15.7336], [83.8821, 42.5759, 21.7609]), 
                          'EX':([116.3348,  56.4458,  16.7164], [82.2490, 42.8175, 22.7280]), 
                          'SE':([115.5042,  56.8636,  18.2710], [81.2206, 42.8957, 23.9444]), 
                          'OD':([116.7937,  56.4556,  16.1290], [82.6207, 42.7140, 21.7858])}

# Hiperparametros:
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
