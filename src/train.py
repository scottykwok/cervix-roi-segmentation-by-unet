import glob
import math
import os
from shutil import copyfile

from PIL import ImageFile
from keras.callbacks import ModelCheckpoint

from configurations import *
from unet import create_model, load_model
from unet_augmentation import getCombinedImageDataGenerator
from util import save_training_history, getTimestamp

if __name__ == "__main__":
    # To allow premature JPG
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if not os.path.exists('./' + info):
        os.makedirs('./' + info)

    UNET_IMAGE_FORMAT = '*.png'

    nbr_train_samples = len(glob.glob(os.path.join(UNET_TRAIN_SPLIT_FOLDER, '*', UNET_IMAGE_FORMAT)))
    nbr_validation_samples = len(glob.glob(os.path.join(UNET_VAL_SPLIT_FOLDER, '*', UNET_IMAGE_FORMAT)))

    # autosave best Model
    best_model_file = os.path.join(info, 'weights.h5')
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

    if os.path.exists(best_model_file):
        print('WARNING: Resume model and weights from previous training ...')
        # Backup previous model file
        copyfile(best_model_file, best_model_file + '.' + getTimestamp())
        model = load_model(img_height, img_width, nb_channels, learning_rate, best_model_file)
        model.summary()
    else:
        print('Using UNET impls  ... save best model to:{}'.format(best_model_file))
        model = create_model(img_height, img_width, nb_channels, learning_rate)
        model.summary()

    steps_per_epoch = math.ceil(1. * nbr_train_samples / batch_size)
    validation_steps = math.ceil(1. * nbr_validation_samples / batch_size)
    print('steps_per_epoch={} , validation_steps={} epochs={}'.format(steps_per_epoch, validation_steps, nbr_epochs))
    if steps_per_epoch <= 0:
        raise AssertionError("Found 0 train samples")
    if validation_steps <= 0:
        raise AssertionError("Found 0 validation samples")


    train_generator = getCombinedImageDataGenerator(
        x_folder=UNET_TRAIN_SPLIT_FOLDER,
        y_folder=UNET_TRAINMASK_SPLIT_FOLDER
    )
    validation_generator = getCombinedImageDataGenerator(
        x_folder=UNET_VAL_SPLIT_FOLDER,
        y_folder=UNET_VALMASK_SPLIT_FOLDER
    )

    print('Start training using ImageDataGenerator:')
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nbr_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[best_model],
        verbose=1)

    save_training_history(info, history)
