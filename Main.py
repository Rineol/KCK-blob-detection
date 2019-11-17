import cv2 as opencv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters, exposure


def treshhold(img):
    x = np.percentile(img, 3.5)
    std = np.std(img)
    if (std < 0.146):
        x -= (10.0 / 255.0)
    else:
        x += (50.0 / 255.0)
    if (img.shape[1] > 1500): x -= (5.0 / 255.0)
    if (std < 0.098): x -= (50.0 / 255.0)
    return (img > x) * 1.0


def increase_contrast(img):
    percmin = 0.3
    percmax = 2.0
    MIN = np.percentile(img, percmin)
    MAX = np.percentile(img, 100 - percmax)
    norm = (img - MIN) / (MAX - MIN)
    norm[norm[:, :] > 1] = 1
    norm[norm[:, :] < 0] = 0
    return norm


def count_blobs(image):
    params = opencv.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minArea = 500
    # Disable the default settings
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = opencv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    im_with_keypoints = opencv.drawKeypoints(
        image, keypoints, np.array([]), (0,0,255),
        opencv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, im_with_keypoints


image = io.imread("Dice-007.jpg", as_gray=True)
image = increase_contrast(image)
image = treshhold(image)
image = (image/256).astype('uint8')
kp, im = count_blobs(image)
print(kp)
print(im)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30, 15))
ax.imshow(image, cmap=plt.cm.gray)
plt.show()


