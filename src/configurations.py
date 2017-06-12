#################
# path config
#################
ROOT_FOLDER = '..'

FILE_PATTERN = '*.jpg'

### Input folders
TRAINSET_INPUT_FOLDER = ROOT_FOLDER + '/input/train'
TESTSET_INPUT_FOLDER = ROOT_FOLDER + '/input/test'
ADDSET_INPUT_FOLDER = ROOT_FOLDER + '/input/additional'

### Output folders
TESTSET_OUTPUT_FOLDER = ROOT_FOLDER + '/input/test_roi'
TRAINSET_OUTPUT_FOLDER = ROOT_FOLDER + '/input/train_roi'
ADDSET_OUTPUT_FOLDER = ROOT_FOLDER + '/input/additional_roi'
OUTPUT_FILE_EXT = '.png'

### How much percentage of extra margin we want to include when cropping the output images
margin = 0.15

### Temp working folders
TRAINSET_RESIZED_FOLDER = ROOT_FOLDER + '/input/train_resized'
TESTSET_RESIZED_FOLDER = ROOT_FOLDER + '/input/test_resized'
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

seed = 20170804
use_symlinks = False  # Can be True if you are on linux
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
