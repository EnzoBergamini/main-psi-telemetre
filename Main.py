import cv2

from Marker import Marker

from MarkerDetector import MarkerDetector

if __name__ == "__main__":
    marker = Marker(1, 200, 5)
    marker_detector = MarkerDetector()

    image = cv2.imread("images/imgCal2.jpg")

    (ids, corners, centers) = marker_detector.detect(image)

    draw_image = marker_detector.draw_markers(ids, corners, centers, image)

    cv2.imshow("image", draw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
