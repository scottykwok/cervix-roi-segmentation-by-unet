from keras.preprocessing.image import ImageDataGenerator

from configurations import *


def getCombinedImageDataGenerator(x_folder, y_folder, debug=False):
    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        shear_range=shear_range,
        zoom_range=zoom_range,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        vertical_flip=vflip,
        horizontal_flip=hflip)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        x_folder,
        class_mode=None,
        seed=seed,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        save_to_dir=VISUAL_RESIZED_FOLDER if debug else None,
        save_prefix='train' if debug else None,
        follow_links=use_symlinks
    )

    mask_generator = mask_datagen.flow_from_directory(
        y_folder,
        class_mode=None,
        seed=seed,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        save_to_dir=VISUAL_RESIZED_FOLDER if debug else None,
        save_prefix='mask' if debug else None,
        follow_links=use_symlinks
    )

    # combine generators into one which yields image and masks
    from itertools import izip
    combined_generator = izip(image_generator, mask_generator)
    return combined_generator
