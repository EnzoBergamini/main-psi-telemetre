import Camera as cam

if __name__ == "__main__":
    camera = cam.Camera("cache-webcam.pickle")
    camera.detect_charuco_corners_live(60)
