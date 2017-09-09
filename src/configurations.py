#################
# path config
#################
ROOT_FOLDER = '..'

FILE_PATTERN = '*.jpg'

OUTPUT_FILE_EXT = '.png'

### Set to True if testset you are predicting stage2 folder
is_stg2 = False

### How much extra margin we want to include when cropping the output images
margin = 0.15
#margin = 0.4   #0.4 seems to work best for my classifier

### Input folders
TRAINSET_INPUT_FOLDER = ROOT_FOLDER + '/input/train'
TESTSET_INPUT_FOLDER = ROOT_FOLDER + '/input/test_stg2' if is_stg2 else ROOT_FOLDER + '/input/test'
ADDSET_INPUT_FOLDER = ROOT_FOLDER + '/input/additional'

### Output folders
TESTSET_OUTPUT_FOLDER = ROOT_FOLDER + '/input/test_stg2_roi_{}'.format(margin) if is_stg2 else ROOT_FOLDER + '/input/test_roi_{}'.format(margin)
TRAINSET_OUTPUT_FOLDER = ROOT_FOLDER + '/input/train_roi_{}'.format(margin)
ADDSET_OUTPUT_FOLDER = ROOT_FOLDER + '/input/additional_roi_{}'.format(margin)


### Temp working folders
TRAINSET_RESIZED_FOLDER = ROOT_FOLDER + '/input/train_resized'
TESTSET_RESIZED_FOLDER = ROOT_FOLDER + '/input/test_stg2_resized' if is_stg2 else ROOT_FOLDER + '/input/test_resized'
ADDSET_RESIZED_FOLDER = ROOT_FOLDER + '/input/additional_resized'
VISUAL_RESIZED_FOLDER = ROOT_FOLDER + '/input/visual_resized'
TRAINSET_RESIZED_MASK_FOLDER = ROOT_FOLDER + '/input/train_resized_mask'

UNET_TRAIN_SPLIT_FOLDER = ROOT_FOLDER + '/input/split_unet/train_split/'
UNET_TRAINMASK_SPLIT_FOLDER = ROOT_FOLDER + '/input/split_unet/train_mask_split/'

UNET_VAL_SPLIT_FOLDER = ROOT_FOLDER + '/input/split_unet/val_split/'
UNET_VALMASK_SPLIT_FOLDER = ROOT_FOLDER + '/input/split_unet/val_mask_split/'

#################
# other parameters
#################
ClassNames = ['Type_1', 'Type_2', 'Type_3']

from sys import platform
use_symlinks = platform == "linux" or platform == "linux2" or platform == "darwin"

seed = 20170804
split_proportion = 0.8

learning_rate = 0.0001
nbr_epochs = 400
batch_size = 32

# Size could be: 64, 80, 144, 128
img_width = 128
img_height = 128
nb_channels = 3

# Augmentation
shear_range = 0.78
zoom_range = 0.4
rotation_range = 180
vflip = True
hflip = True
width_shift_range = 0.3
height_shift_range = 0.3

# preprocessing
rescale = 1. / 255.
preprocessing_function = None

# folder name
info = 'unet' \
       + '_' + str(img_height) + 'x' + str(img_width) + 'x' + str(nb_channels) \
       + '_sp' + str(split_proportion) \
       + '_sh' + str(shear_range) \
       + '_zm' + str(zoom_range) \
       + '_rt' + str(rotation_range) \
       + '_vf' + str(int(vflip)) \
       + '_hf' + str(int(hflip)) \
       + '_ws' + str(width_shift_range) \
       + '_hs' + str(height_shift_range)
