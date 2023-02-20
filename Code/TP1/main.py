import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Gestion des arguments
parser = argparse.ArgumentParser(description="Récupération des éléments presents sur un sol.")
parser.add_argument("--mask", default="", type=str, help="Le mask qui va être pris en compte.")
parser.add_argument("--reference", default="", type=str, help="Le nom de l'image qui va être prise comme référence. Aucun "
                                                         "mask a été appliquer sur cette image.")
parser.add_argument("--img", default="", type=str, help="Le nom de l'image ou les objets peuvent se trouver par terre.")
parser.add_argument("--folder", default="", type=str, help="Le chemin vers le dossier contenant les images.")
args = parser.parse_args()

plt.rcParams["figure.figsize"] = (18, 9)


def equalize_luminosity(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_v = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_v]), cv2.COLOR_HSV2BGR)
    return eq_image


def show_img_with_matplotlib(title, color_img, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    plt.subplot(2, 3, pos)
    plt.imshow(img_RGB, aspect="auto")
    plt.title(title)
    plt.axis('off')


if __name__ == "__main__":
    if args.mask == "" or args.img == "" or args.folder == "" or args.reference == "":
        print("Pas le bon nombre d'argument donne !")
        exit(-1)

    if not os.path.isdir(args.folder):
        print("Le chemin du dossier est incorrect.")
        exit(-1)

    os.chdir(args.folder)

    before = cv2.imread(args.reference)
    after = cv2.imread(args.img)

    before_equalize = equalize_luminosity(before)
    after_equalize = equalize_luminosity(after)
    # Convert images to grayscale
    mask = cv2.cvtColor(cv2.imread(args.mask), cv2.COLOR_BGR2GRAY)
    before_gray = cv2.cvtColor(before_equalize, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_equalize, cv2.COLOR_BGR2GRAY)
    after_gray[mask == 0] = 0
    before_gray[mask == 0] = 0

    diff_box = cv2.subtract(before_gray, after_gray)
    diff_box = cv2.GaussianBlur(diff_box, (5, 5), 0)
    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff_box, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = np.ones((5, 5), np.uint8)
    close = cv2.dilate(thresh, kernel, iterations=5)
    close = cv2.erode(close, kernel, iterations=5)

    contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(filled_after, (x, y), (x + w, y + h), (0, 0, 255), 10)

    show_img_with_matplotlib('Image de référence', before, 1)
    show_img_with_matplotlib('Image à analyser', after, 2)
    show_img_with_matplotlib('Image de la soustraction', cv2.cvtColor(diff_box, cv2.COLOR_GRAY2BGR), 3)
    show_img_with_matplotlib('Image après le threshold', cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), 4)
    show_img_with_matplotlib('Image après la fermeture', cv2.cvtColor(close, cv2.COLOR_GRAY2BGR), 5)
    show_img_with_matplotlib('Image finale', filled_after, 6)
    cv2.imwrite("Resultat.png", filled_after)
    plt.show()
