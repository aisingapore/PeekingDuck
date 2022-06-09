from pathlib import Path
import sys, cv2, time
import numpy as np
from typing import NamedTuple
import argparse

from peekingduck.pipeline.nodes.input import visual
from peekingduck.pipeline.nodes.draw import legend
from peekingduck.pipeline.nodes.output import screen
from peekingduck.pipeline.nodes.dabble import fps

def draw_text(img: np.ndarray, text: str) -> None:
    """ Helper function to draw text on the image """

    x = 15
    y = img.shape[0] - 20

    cv2.putText(
        img = img,
        text = text,
        org = (x, y),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        color = (0, 255, 255), #yellow
        thickness = 2
    )

def calibrate(config):
    camera_source, num_pictures, num_corners, scale_factor, file_path = config.values()

    if camera_source == "0":
        visual_node = visual.Node(source = 0, threading = True)
    else:
        visual_node = visual.Node(source = camera_source, threading = True)

    legend_node = legend.Node(show = ["num_detections", "next_detection_in", "fps_val"])
    screen_node = screen.Node()
    fps_node = fps.Node()

    grid_height = num_corners[0]
    grid_width = num_corners[1]
    num_detections = 0
    last_detection = time.time()

    object_points_base = np.zeros((grid_height * grid_width, 3), np.float32)
    object_points_base[:, :2] = np.mgrid[0:grid_height, 0:grid_width].T.reshape(-1, 2)
    object_points = []
    image_points = []

    # terminal criteria for subpixel finetuning 
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    file_path = Path(file_path) # type: ignore
    # check if file_path has a '.yml' extension
    if file_path.suffix != '.yml':
        raise ValueError("Filepath must have a '.yml' extension.")
    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)

    while True:
        visual_node_outputs = visual_node.run({})
        if visual_node_outputs["pipeline_end"]:
            break
        img = visual_node.run({})["img"]

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detect_corners_success = False

        # if sufficient time has passed
        if time.time() - last_detection >= 5:
            # downscale
            h, w = img.shape[:2]
            new_h = int(h / scale_factor)
            new_w = int(w / scale_factor)

            resized_img = cv2.resize(gray_img, (new_w, new_h), interpolation = cv2.INTER_AREA)

            # try to find chessboard corners
            detect_corners_success, corners = cv2.findChessboardCorners(resized_img, num_corners, None)

            if corners is not None:
                corners = corners * scale_factor

        # cv2 successfully detected the corners
        if detect_corners_success:
            object_points.append(object_points_base)
            image_points.append(corners)

            # improve corner accuracy
            corners_accurate = cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), termination_criteria)

            # draw corners and message on the image
            cv2.drawChessboardCorners(img, num_corners, corners_accurate, detect_corners_success)

            num_detections += 1

            if num_detections != num_pictures:
                draw_text(img, "Detection successful! Press any key to continue.")
            else:
                draw_text(img, "Detection complete! Press any key to exit.")
            
            # display the image and wait for user to press a key
            cv2.imshow("PeekingDuck", img)
            cv2.waitKey(0)

            last_detection = time.time()

            # if we have sufficient images, calculate the coefficients and write to a file
            if num_detections == num_pictures:
                calibration_success, camera_matrix, distortion_coeffs, rotation_vec, translation_vec = cv2.calibrateCamera(object_points, image_points, gray_img.shape[::-1], None, None)

                if calibration_success:
                    print(f"camera_matrix: {camera_matrix}")
                    print(f"distortion_coeffs: {distortion_coeffs}")

                    np.savez(file_path, camera_matrix = camera_matrix, distortion_coeffs = distortion_coeffs)

                    print(f"Calibration successful!")
                else:
                    raise Exception("Calibration failed. Please try again.")

                mean_error = 0
                for i in range(len(object_points)):
                    projected_image_points, _ = cv2.projectPoints(object_points[i], rotation_vec[i], translation_vec[i], camera_matrix, distortion_coeffs)
                    error = cv2.norm(image_points[i], projected_image_points, cv2.NORM_L2) / len(projected_image_points)
                    mean_error += error
                print(f"total eror: {mean_error / len(object_points)}")

                return {"pipeline_end": True}

        # format text for the legend
        num_detections_string = f"{num_detections}/{num_pictures}"

        time_to_next_detection = max(0, int(5 - time.time() + last_detection))
        next_detection_in_string = f"{time_to_next_detection}s"

        fps_val = fps_node.run({"pipeline_end": False})["fps"]

        legend_inputs = {"num_detections": num_detections_string, "next_detection_in": next_detection_in_string, "img": img, "fps_val": fps_val}
        img = legend_node.run(legend_inputs)["img"]
        
        screen_node_outputs = screen_node.run({"img": img})
        if screen_node_outputs["pipeline_end"]:
            break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera_source", type=str, default="0")
    parser.add_argument("-p", "--num_pictures", type=int, default=5)
    parser.add_argument("-d", "--num_corners", type=int, default=[10, 7], nargs = 2)
    parser.add_argument("-s", "--scale_factor", type=int, default=5)
    parser.add_argument("-f", "--file_path", type=str, default="PeekingDuck/data/camera_calibration_coeffs.yml")

    configs = vars(parser.parse_args())
    calibrate(configs)