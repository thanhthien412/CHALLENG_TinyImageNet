import os

TRAIN_IMAGES = "dataset/tiny-imagenet-200/train"
VAL_IMAGES = "dataset/tiny-imagenet-200/val/images"
VAL_MAPPINGS = "dataset/tiny-imagenet-200/val/val_annotations.txt"
WORDNET_IDS = "dataset/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "dataset/tiny-imagenet-200/words.txt"
NUM_CLASSES=200
NUM_TEST_IMAGES=50 * NUM_CLASSES

TRAIN_HDF5='dataset/tiny-imagenet-200/hdf5/train.hdf5'
VAL_HDF5='dataset/tiny-imagenet-200/hdf5/val.hdf5'
TEST_HDF5='dataset/tiny-imagenet-200/hdf5/test.hdf5'

DATASET_MEAN='output/tiny-image-net-200-mean.json'

OUTPUT_PATH='output'


GRAPH_PATH=os.path.join(OUTPUT_PATH,'graph','pic_{}.png')
MODEL_PATH=os.path.join(OUTPUT_PATH,'models','epoch_{}.hdf5')
FIG_PATH = os.path.join(OUTPUT_PATH,"deepergooglenet_tinyimagenet.png")
JSON_PATH = os.path.join(OUTPUT_PATH,"deepergooglenet_tinyimagenet.json")

