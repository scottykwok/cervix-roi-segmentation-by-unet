import json
import ntpath
import os

import cv2
import numpy as np
import glob

from configurations import *


def resize_testset(source_folder, target_folder, dsize, pattern=FILE_PATTERN):
    print('Resizing testset ...')
    if not os.path.exists(target_folder): os.makedirs(target_folder)
    total_images = glob.glob(os.path.join(source_folder, pattern))
    total = len(total_images)
    for i, source in enumerate(total_images):
        filename = ntpath.basename(source)
        target = os.path.join(target_folder, filename.replace('.jpg', '.png'))

        img = cv2.imread(source)
        img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(target, img_resized)
        if i % 100 == 0:
            print("Resized {}/{} images".format(i, total))


def resize_addset(source_folder, target_folder, dsize, pattern=FILE_PATTERN):
    print('Resizing additional set...')
    if not os.path.exists(target_folder): os.makedirs(target_folder)
    for clazz in ClassNames:
        if clazz not in os.listdir(target_folder):
            os.makedirs(os.path.join(target_folder, clazz))

        total_images = glob.glob(os.path.join(source_folder, clazz, pattern))
        total = len(total_images)
        for i, source in enumerate(total_images):
            filename = ntpath.basename(source)
            target = os.path.join(target_folder, clazz, filename.replace('.jpg', '.png'))

            try:
                img = cv2.imread(source)
                img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(target, img_resized)
            except:
                print('-------------------> error in: {}'.format(source))

            if i % 20 == 0:
                print("Resized {}/{} images".format(i, total))


def resize_trainset_and_generate_masks(dsize):
    print('Resizing train set & creating masks ...')
    INPUT_FOLDER = ROOT_FOLDER + '/input'
    annotation_json_filename = INPUT_FOLDER + '/{}_bbox.json'
    for c in ClassNames:

        annotation_json = annotation_json_filename.format(c)
        print('Loading {}'.format(annotation_json))
        with open(annotation_json) as json_file:
            data = json.load(json_file)

        OUTPUT_FOLDER = os.path.join(TRAINSET_RESIZED_MASK_FOLDER, c)
        if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

        OUTPUT2_FOLDER = os.path.join(TRAINSET_RESIZED_FOLDER, c)
        if not os.path.exists(OUTPUT2_FOLDER): os.makedirs(OUTPUT2_FOLDER)

        total = len(data)
        for i, row in enumerate(data):
            if i % 10 == 0:
                print('Processing {}/{}'.format(i, total))

            input_filename = os.path.join(INPUT_FOLDER, row['filename'])
            if not os.path.exists(input_filename):
                print('Skipped input file not exist: {}'.format(input_filename))
                continue

            basename = ntpath.basename(input_filename)
            output_filename = os.path.join(OUTPUT_FOLDER, basename.replace('.jpg', '.png'))
            if os.path.exists(output_filename):
                print('Skipped output already exist: {}'.format(output_filename))
                continue

            output2_filename = os.path.join(OUTPUT2_FOLDER, basename.replace('.jpg', '.png'))
            if os.path.exists(output2_filename):
                print('Skipped output already exist: {}'.format(output2_filename))
                continue

            annotation = row['annotations'][0]
            x = int(round(annotation['x']))
            y = int(round(annotation['y']))
            w = int(round(annotation['width']))
            h = int(round(annotation['height']))
            img = cv2.imread(input_filename)

            resized = cv2.resize(img, dsize=dsize)
            cv2.imwrite(output2_filename, resized)

            mask = np.zeros_like(img)
            mask[y:y + h, x:x + w, :] = 255

            mask = cv2.resize(mask, dsize=dsize)
            cv2.imwrite(output_filename, mask)


if __name__ == '__main__':
    resize_trainset_and_generate_masks(dsize=(img_height, img_width))

    resize_testset(TESTSET_INPUT_FOLDER, TESTSET_RESIZED_FOLDER, (img_width, img_height))

    if os.path.exists(ADDSET_INPUT_FOLDER):
        resize_addset(ADDSET_INPUT_FOLDER, ADDSET_RESIZED_FOLDER, (img_width, img_height))
