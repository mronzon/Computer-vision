import argparse
import os
import cv2
from os import walk

parser = argparse.ArgumentParser(description="Récupération des éléments presents sur un sol.")
parser.add_argument("--folder", default="", type=str, help="Le chemin vers le dossier contenant les images.")
args = parser.parse_args()

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

    for filename in filenames:
        image = cv2.imread(filename)  # Read the image
        equalized = equalize_luminosity(image)  # Equalize the luminosity
        gray_equalized = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)  # Convert to gray scale

        thresh_gray_equalized = cv2.threshold(gray_equalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Do a treshold to supress the blur
        cv2.imwrite(result_folder + "\\" + os.path.splitext(filename)[0] + "_gray_equalized" + os.path.splitext(filename)[1], thresh_gray_equalized)  # Save the image
