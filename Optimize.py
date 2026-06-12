import cv2
import numpy as np


def dispose_img(dilate):

    cout_white = []

    for ii in range(0, 640):
        sum_white = 0
        for jj in range(0, 480):
            if dilate[jj][ii] == 255:
                sum_white += 1
        cout_white.append(sum_white)
    img_mean = np.mean(cout_white)
    img_each = np.std(cout_white)
    # print(img_mean, img_each)
    sum2 = img_mean + 2 * img_each
    # print(sum2, img_mean, img_each)
    for k in range(0, 640):
        if cout_white[k] != 0:
            if cout_white[k] >= sum2 or cout_white[k] > 240:
                dilate[:, k] = 255
            else:
                dilate[:, k] = 0
            if cout_white[k] <= 10:
                dilate[:, k] = 0

    return dilate


def filter_continuous_columns(dilate):

    white_cols = np.all(dilate == 255, axis=0)

    changes = np.diff(white_cols.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if white_cols[0]:
        starts = np.insert(starts, 0, 0)
    if white_cols[-1]:
        ends = np.append(ends, len(white_cols))

    for start, end in zip(starts, ends):
        if end - start < 18:
            dilate[:, start:end] = 0

    return dilate
