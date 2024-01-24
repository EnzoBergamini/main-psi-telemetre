import cv2
import numpy as np


class MarkerDetector(cv2.aruco.ArucoDetector):
    def __init__(self):
        super().__init__(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50),
            cv2.aruco.DetectorParameters(),
        )

    def detect(self, image) -> tuple:
        (corners, ids, rejected) = self.detectMarkers(image)

        new_ids = []
        new_centers = []
        new_corners = []

        print("Il y a ", len(corners), "marker aruco \n")

        if len(corners) > 0:
            ids = ids.flatten()
            for markerCorner, markerID in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                new_ids.append(markerID)
                new_centers.append((cX, cY))
                new_corners.append((topLeft, topRight, bottomRight, bottomLeft))

                # draw the ArUco marker ID on the image
                print(
                    "[INFO] ArUco marker ID: {} aux coordonnées ({}, {})".format(
                        markerID, cX, cY
                    )
                )
        return new_ids, new_corners, new_centers

    def draw_markers(self, ids, corners, centers, image):
        if len(centers) > 0:
            for corner, center in zip(corners, centers):
                # Définir la longueur des axes
                axis_length = 50

                # Centre du marqueur
                cX, cY = center

                # Dessiner l'axe X en rouge
                cv2.line(image, (cX, cY), (cX + axis_length, cY), (0, 0, 255), 2)

                # Dessiner l'axe Y en vert
                cv2.line(image, (cX, cY), (cX, cY + axis_length), (0, 255, 0), 2)

                # Optionnel : Dessiner le contour du marqueur
                cv2.polylines(
                    image, [np.array(corner, dtype=np.int32)], True, (255, 0, 0), 2
                )

        return image
