import json
import ntpath
import os

import cv2
import numpy as np

from configurations import *

if __name__ == '__main__':
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
            output_filename = os.path.join(OUTPUT_FOLDER, basename)
            if os.path.exists(output_filename):
                print('Skipped output already exist: {}'.format(output_filename))
                continue

            output2_filename = os.path.join(OUTPUT2_FOLDER, basename)
            if os.path.exists(output2_filename):
                print('Skipped output already exist: {}'.format(output2_filename))
                continue

            annotation = row['annotations'][0]
            x = int(round(annotation['x']))
            y = int(round(annotation['y']))
            w = int(round(annotation['width']))
            h = int(round(annotation['height']))
            img = cv2.imread(input_filename)

            resized = cv2.resize(img, dsize=(img_height, img_width))
            cv2.imwrite(output2_filename, resized)

            mask = np.zeros_like(img)
            mask[y:y + h, x:x + w, :] = 255

            mask = cv2.resize(mask, dsize=(img_height, img_width))
            cv2.imwrite(output_filename, mask)
