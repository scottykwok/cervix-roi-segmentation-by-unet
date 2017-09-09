#Code taken from https://www.kaggle.com/chattob/cervix-segmentation-gmm

import glob
import ntpath
import os

import cv2
import numpy as np

from configurations import *


def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else:
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i - position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif (height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area = maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea


def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r - 1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (
        int(maxArea[3] + 1 - maxArea[0] / abs(maxArea[1] - maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])


def cropCircle(img, resize=None):
    if resize:
        if (img.shape[0] > img.shape[1]):
            tile_size = (int(img.shape[1] * resize / img.shape[0]), resize)
        else:
            tile_size = (resize, int(img.shape[0] * resize / img.shape[1]))
        img = cv2.resize(img, dsize=tile_size, interpolation=cv2.INTER_CUBIC)
    else:
        tile_size = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    ff = np.zeros((gray.shape[0], gray.shape[1]), 'uint8')
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1] / 2), int(gray.shape[0] / 2)), 1)

    rect = maxRect(ff)
    rectangle = [min(rect[0], rect[2]), max(rect[0], rect[2]), min(rect[1], rect[3]), max(rect[1], rect[3])]
    img_crop = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
    cv2.rectangle(ff, (min(rect[1], rect[3]), min(rect[0], rect[2])), (max(rect[1], rect[3]), max(rect[0], rect[2])), 3,
                  2)

    return [img_crop, rectangle, tile_size]


if __name__ == '__main__':

    #### TRAIN SET

    # INPUT_FOLDER = ROOT_FOLDER + '/input/train'
    # CROPSET_FOLDER = ROOT_FOLDER + '/input/train_cropped'
    #
    # total_images = glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN))
    # total = len(total_images)
    #
    # for clazz in ClassNames:
    #     OUTPUT_FOLDER = os.path.join(CROPSET_FOLDER, clazz)
    #     if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    #
    #     total_images = glob.glob(os.path.join(INPUT_FOLDER, clazz, FILE_PATTERN))
    #     total = len(total_images)
    #     for i, input_filename in enumerate(total_images):
    #         img = cv2.imread(input_filename)
    #
    #         img_crop, rectangle, tile_size = cropCircle(img, resize=None)
    #
    #         basename = ntpath.basename(input_filename)
    #         output_filename = os.path.join(OUTPUT_FOLDER, basename)
    #         cv2.imwrite(output_filename, img_crop)
    #
    #         if i % 20 == 0:
    #             print("Cropped {}/{} images".format(i, total))
    #

    INPUT_FOLDER = ROOT_FOLDER + '/input/test'
    CROPSET_FOLDER = ROOT_FOLDER + '/input/test_cropped'

    total_images = glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN))
    total = len(total_images)

    OUTPUT_FOLDER = CROPSET_FOLDER
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    total_images = glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN))
    total = len(total_images)
    for i, input_filename in enumerate(total_images):
        img = cv2.imread(input_filename)

        img_crop, rectangle, tile_size = cropCircle(img, resize=None)

        basename = ntpath.basename(input_filename)
        output_filename = os.path.join(OUTPUT_FOLDER, basename)
        cv2.imwrite(output_filename, img_crop)

        if i % 20 == 0:
            print("Cropped {}/{} images".format(i, total))
