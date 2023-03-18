import argparse
import os
import cv2
from skimage import img_as_float, img_as_ubyte
from skimage.feature import peak_local_max
from skimage.io import imsave
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
from os import walk
from skimage.segmentation import slic, watershed

parser = argparse.ArgumentParser(description="Segmentations d'une image remplis de caillou en petite images possédant un caillou à chaque fois.")
parser.add_argument("--folder", default="", type=str, help="Le chemin vers le dossier contenant les images.")
args = parser.parse_args()


def crop_image(img, label):
    result = img.copy()
    min_y = len(label)
    min_x = len(label)
    max_x = -1
    max_y = -1
    for y in range(len(label)):
        for x in range(len(label[y])):
            if label[y][x] != 0:
                min_x = min(x, min_x)
                min_y = min(y, min_y)
                max_x = max(x, max_x)
                max_y = max(y, max_y)
    return result[min_y:max_y, min_x:max_x]


def black_proportion(image):
    total = 0
    black = 0
    for y in image:
        for x in y:
            total += 1
            if x.all() == 0:
                black += 1
    return black / total


def equalize_luminosity(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_v = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_v]), cv2.COLOR_HSV2BGR)
    return eq_image


if __name__ == "__main__":
    result_folder = "Result"

    if not os.path.isdir(args.folder):
        print("Le chemin du dossier est incorrect.")
        exit(-1)

    os.chdir(args.folder)

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    _, _, filenames = next(walk(args.folder))
    i = 0
    slic_data = {"Moyenne de B": [], "Moyenne de G": [], "Moyenne de R": []}
    index = []
    for filename in filenames:
        image = cv2.imread(filename)  # Read the image
        equalized = equalize_luminosity(image)  # Equalize the luminosity
        gray_equalized = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)  # Convert to gray scale
        thresh_gray_equalized = cv2.threshold(gray_equalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        equalized[thresh_gray_equalized == 0] = 0
        img = img_as_float(equalized)

        segments_slic = slic(img, n_segments=100, compactness=10, sigma=1)
        unique_label = np.unique(segments_slic)

        for label in unique_label:
            segment_label = segments_slic.copy()
            segment_label[segment_label != label] = 0
            result = crop_image(image, segment_label)
            if black_proportion(result) > 0.5:
                continue
            imsave(result_folder + "\\rock_" + str(i) + os.path.splitext(filename)[1], img_as_ubyte(result))  # Save the image
            i += 1
            # Creation of the DataFrame.
            index.append(f"Grain isolé {i}")
            slic_data["Moyenne de B"].append(np.average(result[0, :, :]))
            slic_data["Moyenne de G"].append(np.average(result[:, 0, :]))
            slic_data["Moyenne de R"].append(np.average(result[:, :, 0]))

    dataFrame_slic = pd.DataFrame(slic_data, index=index)
    print(dataFrame_slic)

