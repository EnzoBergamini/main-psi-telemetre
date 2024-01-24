""" The module contains camera related classes and functions. """

import pickle
import cv2 as cv
from cv2 import aruco
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any


class Camera:
    """The class implements the camera."""

    def __init__(self, cache_file: str):
        self.mtx = None
        self.dist = None
        self._calibrated = False
        self.cache_file = cache_file

        if Path(self.cache_file).exists():
            self._load()
            self._calibrated = True

    @property
    def calibrated(self) -> bool:
        """The property is equal true if the camera calibrated"""
        return self._calibrated

    def undistort(self, img: Any):
        """Returns undistorted image. Takes gray scaled image as an input."""
        return cv.undistort(img, self.mtx, self.dist, None, self.mtx)

    def calibrate_with_chessboard(
        self, images: List[str], pattern_size: Tuple[int, int], output: str = None
    ) -> None:
        """Calibrate the camera with set of images with the specified chessboard"""
        w, h = pattern_size
        obj_points = []  # 3D real word points
        img_points = []  # 2D image points
        obj_coords = np.zeros((w * h, 3), np.float32)
        obj_coords[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        gray = None

        for file in images:
            img = cv.imread(file)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            res, corners = cv.findChessboardCorners(gray, pattern_size, None)

            if not res:
                print(f"Unable to find chessboard corners on {file}")
                continue

            if output:
                img = cv.drawChessboardCorners(img, pattern_size, corners, res)
                if not cv.imwrite(f"{output}/{Path(file).name}", img):
                    raise IOError(f"Unable to store the debug output for {file}")

            img_points.append(corners)
            obj_points.append(obj_coords)

        shape = gray.shape[::-1]
        _, self.mtx, self.dist, _, _ = cv.calibrateCamera(
            obj_points, img_points, shape, None, None
        )
        self._calibrated = True
        self._save()

    def detect_charuco_corners(self, video_in: str, video_out: str) -> None:
        """Detect charuco corners on video and overlay an additional information"""
        cap = cv.VideoCapture(video_in)
        board = Camera._create_charuco_board()
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        output = cv.VideoWriter(
            video_out,
            cv.VideoWriter_fourcc("M", "J", "P", "G"),
            30,
            (frame_width, frame_height),
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, Camera._charuco_dict())
            cv.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
            out = frame.copy()
            if corners:
                cv.aruco.drawDetectedMarkers(out, corners, ids, borderColor=(0, 255, 0))
                res, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board, self.mtx, self.dist
                )
                if ret:
                    cv.aruco.drawDetectedCornersCharuco(
                        out, charuco_corners, charuco_ids, cornerColor=(0, 0, 255)
                    )
                rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                    corners, 0.02, self.mtx, self.dist
                )
                for i in range(len(tvecs)):
                    cv.drawFrameAxes(out, self.mtx, self.dist, rvecs[i], tvecs[i], 0.03)
            output.write(out)

        cap.release()
        output.release()

    def calibrate_with_charuco_live(self, frame_skip: int = 30) -> None:
        """Calibrate the camera live with a ChArUco board."""

        cap = cv.VideoCapture(0)  # 0 for default webcam
        board = Camera._create_charuco_board()
        _dict = Camera._charuco_dict()
        all_corners = []
        all_ids = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip == 0:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                corners, ids, _ = cv.aruco.detectMarkers(gray, _dict)
                if corners:
                    (
                        res,
                        charuco_corners,
                        charuco_ids,
                    ) = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    if res > 2:  # Adjust the threshold as needed
                        all_corners.append(charuco_corners)
                        all_ids.append(charuco_ids)
                        frame = cv.aruco.drawDetectedMarkers(frame, corners, ids)
                        frame = cv.aruco.drawDetectedCornersCharuco(
                            frame, charuco_corners, charuco_ids
                        )

                cv.imshow("Calibration", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to stop
                    break

        cap.release()
        cv.destroyAllWindows()

        if len(all_corners) > 0:
            shape = gray.shape[::-1]
            _, self.mtx, self.dist, _, _ = cv.aruco.calibrateCameraCharuco(
                all_corners, all_ids, board, shape, None, None
            )
            self._calibrated = True
            self._save()

    def calibrate_with_charuco(self, video_sample: str, frame_rate: int) -> None:
        """Calibrate the camera with a video sample with ChArUco board DICT_7X7_50"""
        cap = cv.VideoCapture(video_sample)
        _dict = Camera._charuco_dict()
        board = Camera._create_charuco_board()
        all_corners = []
        all_ids = []
        frame_id = 0
        gray = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_rate != 0:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _ = cv.aruco.detectMarkers(gray, _dict)
            if corners:
                res, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                if res > 20:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)

        _, self.mtx, self.dist, _, _ = cv.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, gray.shape, None, None
        )
        cap.release()
        self._calibrated = True
        self._save()
        cv.destroyAllWindows()

    def calibrate_with_chessboard_live(
        self, pattern_size: Tuple[int, int], frame_skip: int = 30
    ) -> None:
        w, h = pattern_size
        obj_points, img_points = [], []
        obj_coords = np.zeros((w * h, 3), np.float32)
        obj_coords[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        cap = cv.VideoCapture(0)  # 0 for default camera
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip == 0:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                res, corners = cv.findChessboardCorners(gray, pattern_size, None)

                if res:
                    img_points.append(corners)
                    obj_points.append(obj_coords)
                    img = cv.drawChessboardCorners(frame, pattern_size, corners, res)
                    cv.imshow("Calibration", img)

                    if cv.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to stop
                        break

        cap.release()
        cv.destroyAllWindows()

        if len(obj_points) > 0:
            shape = gray.shape[::-1]
            _, self.mtx, self.dist, _, _ = cv.calibrateCamera(
                obj_points, img_points, shape, None, None
            )
            self._calibrated = True
            self._save()

    def detect_charuco_corners_live(self, frame_rate: int = 30) -> None:
        cap = cv.VideoCapture(0)
        board = Camera._create_charuco_board()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, Camera._charuco_dict())
            cv.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
            out = frame.copy()
            if corners:
                cv.aruco.drawDetectedMarkers(out, corners, ids, borderColor=(0, 255, 0))
                res, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board, self.mtx, self.dist
                )
                if res:
                    cv.aruco.drawDetectedCornersCharuco(
                        out, charuco_corners, charuco_ids, cornerColor=(0, 0, 255)
                    )
                    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                        corners, 0.02, self.mtx, self.dist
                    )
                    for i in range(len(tvecs)):
                        cv.drawFrameAxes(
                            out, self.mtx, self.dist, rvecs[i], tvecs[i], 0.03
                        )
            cv.imshow("ChArUco Corners", out)

            if cv.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
                break

        cap.release()
        cv.destroyAllWindows()

    def _save(self):
        data = {
            "mtx": self.mtx,
            "dist": self.dist,
        }
        pickle.dump(data, open(self.cache_file, "wb"))

    def _load(self):
        data = pickle.load(open(self.cache_file, "rb"))
        self.mtx = data["mtx"]
        self.dist = data["dist"]

    @staticmethod
    def _create_charuco_board():
        return cv.aruco.CharucoBoard((7, 7), 0.04, 0.02, Camera._charuco_dict())

    @staticmethod
    def _charuco_dict():
        return cv.aruco.getPredefinedDictionary(cv.aruco.DICT_7X7_50)
