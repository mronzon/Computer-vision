import argparse
import os
from os import walk

import tensorflow as tf
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

parser = argparse.ArgumentParser(description="Comparaison d'un CNN classique avec un modèle pré-entraîné.")
parser.add_argument("--folder", default="", type=str, help="Le chemin vers le dossier contenant le jeu de données.")
parser.add_argument("--input", default=(0, 0, 0), type=tuple, help="la taille désirée des images")
parser.add_argument("--model", default="", type=str, help="Le modèle à entraîner")
args = parser.parse_args()


if __name__ == "__main__":

    if not os.path.isdir(args.folder):
        print("Le chemin du dossier est incorrect.")
        exit(-1)

    train_dir = args.folder + "\\train"
    val_dir = args.folder + "\\validation"

    _, dir_names, _ = next(walk(train_dir))

    num_classes = len(dir_names)
    input_shape = args.input

    # Créer un objet ImageDataGenerator avec les augmentations d'images souhaitées
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    # Utiliser l'objet ImageDataGenerator pour charger les données d'entraînement
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(100, 100),
                                                        batch_size=32,
                                                        class_mode='categorical')

    # Utiliser l'objet ImageDataGenerator pour charger les données de validation
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(val_dir,
                                                    target_size=(100, 100),
                                                    batch_size=32,
                                                    class_mode='categorical')
    model = ""
    if args.modele == "CNN":
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, kernel_size=(3, 3), activation="relu"),
                Flatten(),
                Dropout(0.5),
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.summary()
    elif args.modele == "VGG16":
        # Chargement de VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        # Construction du modèle
        model = Sequential([
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        # Congélation des couches de VGG16
        for layer in base_model.layers:
            layer.trainable = False

    if model == "":
        print("Le nom du modèle n'est pas le bon")
        exit(-1)

    # Compilation du modèle
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(train_generator, validation_data=val_generator, epochs=15)

