def cv2_morph_close(binary_image, size=5):
    import cv2
    from skimage.morphology import disk
    kernel = disk(size)
    result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return result


def cv2_morph_open(binary_image, size=5):
    import cv2
    from skimage.morphology import disk
    kernel = disk(size)
    result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return result


def morphology_clean(mask_binary):
    return cv2_morph_close(cv2_morph_open(mask_binary))


def getTimestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_training_history(info, history):
    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.gcf().savefig('./' + info + '/loss_history.' + getTimestamp() + '.jpg')
    # plt.show()

    # summarize history for dice_coef
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice_coef')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.gcf().savefig('./' + info + '/dice_coef_history.' + getTimestamp() + '.jpg')
    # plt.show()

    # history to json file
    import json

    with open('./' + info + '/log.' + getTimestamp() + '.json', 'w') as fp:
        json.dump(history.history, fp, indent=True)
