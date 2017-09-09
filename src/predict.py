import glob
import math
import ntpath
import os

import cv2
import numpy as np

from configurations import *
from unet import load_model
from util import *


def preprocessing(img):
    return img * rescale


def inverse_preprocessing(img):
    return img / rescale


def to_binary_mask(mask, t=0.00001):
    mask = inverse_preprocessing(mask)

    ### Threshold the RGB image  - This step increase sensitivity
    mask[mask > t] = 255
    mask[mask <= t] = 0

    ### To grayscale and normalize
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray = cv2.normalize(src=mask_gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    ### Auto binary threshold
    (thresh, mask_binary) = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask_binary


def find_bbox(mask_binary, margin_factor=None):
    ret, thresh = cv2.threshold(mask_binary, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) == 0:
        return (0, 0, mask_binary.shape[0], mask_binary.shape[1], False)
    else:
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        x, y, w, h = cv2.boundingRect(cnt)

        if margin_factor != None and margin_factor > 0:
            wm = w * margin_factor
            hm = h * margin_factor
            x -= wm
            y -= hm
            w += 2 * wm
            h += 2 * hm
            x = max(0, x)
            y = max(0, y)
            X = min(x + w, mask_binary.shape[1])
            Y = min(y + h, mask_binary.shape[0])
            w = X - x
            h = Y - y
        return (int(x), int(y), int(w), int(h), True)


def transform_bbox(bbox, from_dim, to_dim):
    H0, W0 = from_dim
    H1, W1 = to_dim
    x, y, w, h = bbox
    w_factor = 1. * W1 / W0
    h_factor = 1. * H1 / H0
    return max(0, int(math.floor(x * w_factor))), \
           max(0, int(math.floor(y * h_factor))), \
           int(math.floor(w * w_factor)), \
           int(math.floor(h * h_factor))


def predict_and_crop(model, original_folder, resized_folder, output_folder, margin_factor):
    generate_previews = False #Set to True if you want to see the overlay of bbox on original image
    generate_crops = True
    generate_masks = False

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # Test images
    print('Input folder: {}'.format(resized_folder))
    test_image_files = np.sort(glob.glob(os.path.join(resized_folder, '*.png')))
    total = len(test_image_files)
    for i, filename in enumerate(test_image_files):
        if i > 0 and i % 50 == 0:
            print('Processed {}/{} files ...'.format(i, total))

        basename = ntpath.basename(filename)
        img1 = cv2.resize(cv2.imread(filename), dsize=(img_height, img_width))
        img = preprocessing(img1)
        img = img[None,]  # Add dimension

        predict = model.predict(img, batch_size=1, verbose=0)

        # extract binary mask
        binary_mask = to_binary_mask(predict[0])
        morphed_mask = morphology_clean(binary_mask)
        x, y, w, h, success = find_bbox(morphed_mask, margin_factor)

        original_img_file = os.path.join(original_folder, basename.replace('.png', '.jpg'))
        original = cv2.imread(original_img_file)
        if original is None:
            raise AssertionError("Cannot read the original image:{}".format(original_img_file))

        # transform bbox back to original dimension
        x1, y1, w1, h1 = transform_bbox(bbox=(x, y, w, h), from_dim=morphed_mask.shape, to_dim=original.shape[0:2])

        if generate_crops:
            cropped = original[y1:y1 + h1, x1:x1 + w1, :]
            cropped_filename = os.path.join(output_folder, basename.replace('.png', OUTPUT_FILE_EXT))
            if cropped.mean() <= 15 or not success: # a black crop or fail to find bounding box
                from crop import cropCircle
                img_crop, rectangle, tile_size = cropCircle(original, resize=None)
                cv2.imwrite(cropped_filename, img_crop)
            else:
                cv2.imwrite(cropped_filename, cropped)

        # For debug & preview
        if generate_masks:
            cv2.imwrite(os.path.join(output_folder, basename.replace('.png', '_mask.png')), morphed_mask)

        if generate_previews:
            # Highlight the mask in original
            img_highlighted = original.copy()
            original_mask = cv2.resize(morphed_mask, dsize=(original.shape[1], original.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            blue_channel = img_highlighted[:, :, 0]
            blue_channel[original_mask > 0] = 255
            cv2.rectangle(img_highlighted, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
            preview_filename = os.path.join(output_folder, basename.replace('.png', '_preview.jpg'))
            cv2.imwrite(preview_filename, img_highlighted)


if __name__ == '__main__':
    weight_file = os.path.join(info, 'weights.h5')
    model = load_model(img_height, img_width, nb_channels, learning_rate, weight_file)

    # predict the ROI of test images
    predict_and_crop(model, TESTSET_INPUT_FOLDER, TESTSET_RESIZED_FOLDER, TESTSET_OUTPUT_FOLDER, margin)

    # predict the ROI of train images
    for c in ClassNames:
        ORIGINAL_FOLDER = os.path.join(TRAINSET_INPUT_FOLDER, c)
        INPUT_FOLDER = os.path.join(TRAINSET_RESIZED_FOLDER, c)
        OUTPUT_FOLDER = os.path.join(TRAINSET_OUTPUT_FOLDER, c)
        predict_and_crop(model, ORIGINAL_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER, margin)

    if os.path.exists(ADDSET_INPUT_FOLDER):
        # predict the ROI of additional images
        for c in ClassNames:
            ORIGINAL_FOLDER = os.path.join(ADDSET_INPUT_FOLDER, c)
            INPUT_FOLDER = os.path.join(ADDSET_RESIZED_FOLDER, c)
            OUTPUT_FOLDER = os.path.join(ADDSET_OUTPUT_FOLDER, c)
            predict_and_crop(model, ORIGINAL_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER, margin)
