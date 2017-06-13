import glob
import ntpath
import os
import shutil

import numpy as np

from configurations import *


def split_folder(seed, targe_prefix, clazz, total_images, split_proportion, train_split_folder, val_split_folder):
    np.random.seed(seed)
    nbr_train = int(len(total_images) * split_proportion)
    np.random.shuffle(total_images)
    train_images = total_images[:nbr_train]
    val_images = total_images[nbr_train:]

    # create split for TRAIN
    if not os.path.exists(train_split_folder): os.makedirs(train_split_folder)
    if clazz not in os.listdir(train_split_folder):
        os.mkdir(os.path.join(train_split_folder, clazz))

    for source in train_images:
        filename = ntpath.basename(source)
        target = os.path.join(train_split_folder, clazz, targe_prefix + filename)
        if use_symlinks:
            absSource = os.path.abspath(source)
            absTarget = os.path.abspath(target)
            os.symlink(absSource, absTarget)
        else:
            shutil.copy(source, target)

    # create split for VAL
    if not os.path.exists(val_split_folder): os.makedirs(val_split_folder)
    if clazz not in os.listdir(val_split_folder):
        os.mkdir(os.path.join(val_split_folder, clazz))

    for source in val_images:
        filename = ntpath.basename(source)
        target = os.path.join(val_split_folder, clazz, targe_prefix + filename)
        if use_symlinks:
            absSource = os.path.abspath(source)
            absTarget = os.path.abspath(target)
            os.symlink(absSource, absTarget)
        else:
            shutil.copy(source, target)


if __name__ == '__main__':
    # Png is better
    UNET_INPUT_FILE_PATTERN = '*.png'

    # Split train set
    for clazz in ClassNames:
        total_images = np.sort(glob.glob(os.path.join(TRAINSET_RESIZED_FOLDER, clazz, UNET_INPUT_FILE_PATTERN)))
        split_folder(seed, '', clazz, total_images, split_proportion, UNET_TRAIN_SPLIT_FOLDER, UNET_VAL_SPLIT_FOLDER)
    print('Finish splitting train and val set')

    # Split mask set
    for clazz in ClassNames:
        total_images = np.sort(glob.glob(os.path.join(TRAINSET_RESIZED_MASK_FOLDER, clazz, UNET_INPUT_FILE_PATTERN)))
        split_folder(seed, '', clazz, total_images, split_proportion, UNET_TRAINMASK_SPLIT_FOLDER,
                     UNET_VALMASK_SPLIT_FOLDER)
    print('Finish splitting train_mask and val_mask set')

    for clazz in ClassNames:
        total_images_A = np.sort(glob.glob(os.path.join(TRAINSET_RESIZED_FOLDER, clazz, UNET_INPUT_FILE_PATTERN)))
        total_images_B = np.sort(glob.glob(os.path.join(TRAINSET_RESIZED_MASK_FOLDER, clazz, UNET_INPUT_FILE_PATTERN)))
        print('{}: {} trains vs {} masks'.format(clazz, len(total_images_A), len(total_images_B)))
        assert len(total_images_A) == len(total_images_B)

    nb_train_images = len(glob.glob(os.path.join(UNET_TRAIN_SPLIT_FOLDER, '*', UNET_INPUT_FILE_PATTERN)))
    nb_val_images = len(glob.glob(os.path.join(UNET_VAL_SPLIT_FOLDER, '*', UNET_INPUT_FILE_PATTERN)))
    print('No. of train data: {}'.format(nb_train_images))
    print('No. of val   data: {}'.format(nb_val_images))
