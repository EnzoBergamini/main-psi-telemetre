import cv2
import numpy as np
import matplotlib.pyplot as plt


class Marker:
    def __init__(self, id, size, border_size):
        self.__id = id
        self.__size = size
        self.__border_size = border_size
        self.__dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
        self.__marker = np.zeros((self.__size, self.__size, 1), dtype=np.uint8)
        cv2.aruco.generateImageMarker(
            self.__dictionary, id, size, self.__marker, border_size
        )

    def __str__(self):
        return "Marker(id={}, size={}, border_size={}, dictionary={})".format(
            self.__id, self.__size, self.__border_size, self.__dictionary
        )

    def display(self):
        plt.imshow(self.__marker, cmap="gray", interpolation="nearest")
        plt.show()

    def get_marker(self):
        return self.__marker
