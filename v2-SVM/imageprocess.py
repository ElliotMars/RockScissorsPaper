import cv2
from skimage.feature import hog
import numpy as np


def augment_image(img):
    augmented_images = [img]

    # 翻转图像
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 旋转图像
    for angle in [90, 180, 270]:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotated)

    return augmented_images


def augment_images(images):
    augmented_images = []
    for img in images:
        augmented_images.extend(augment_image(img))
    return augmented_images


def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fd = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), visualize=False)
        hog_features.append(fd)
    return hog_features
