import numpy as np
from contour_solution import *
from geometry import *
from homography import *
from common import *
import cv2
import time




# 程序的开始
def start_camera():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)

    # 获取单应性矩阵

    Homography = np.array([[1.14109173e-01, -3.15012432e+00, 1.24980885e+03],
                           [-2.12689088e-02, 2.08495473e-01, -1.27508350e+01],
                           [1.02465396e-05, -2.45349184e-03, 1.00000000e+00]])

    # Homography = get_Homography(camera)

    print(Homography)

    Homography_inv = np.matrix(Homography)
    cv2.namedWindow('background', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('background', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    h, w = get_scree_wh()

    background = np.ones((w, h, 3), dtype=np.float32)

    run_num = 30
    average_position = [-1, -1]
    last_average_position = [-1, -1]
    # button_list = []
    # button_list = [[h / 4, w / 2], [h / 4 * 3, w / 2]]
    # button_size = 600

    average_positions = []

    t = time.time()

    while True:
        point_list = []
        for i in range(run_num):
            # T1 = time.time()
            # 读取这一帧的图像
            ret, frame = camera.read()

            # !!!
            # frame = cv2.imread('test.jpg')
            # frame = cv2.resize(frame, (640, 480))

            # 图像反转
            frame_flip = cv2.flip(frame, 1)

            position, is_get_position = use_contour_to_get_position(frame_flip)

            background_copy = background.copy()

            if is_get_position:
                cv2.circle(frame_flip, position, 5, (255, 0, 0), -1, cv2.LINE_AA)
                scree_position = use_Homography_transform_point(position, Homography_inv)
                cv2.putText(background_copy, str(average_position), (int(h / 10), int(w / 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.circle(background_copy, (int(average_position[0]), int(average_position[1])), 40, (255, 0, 255), 15,
                           cv2.LINE_AA)

                point_list.append([scree_position[0], scree_position[1]])

            if average_position[0] > 0 and average_position[1] > 0 and last_average_position[0] > 0 and \
                    last_average_position[1] > 0:
                cv2.line(background_copy, [int(average_position[0]), int(average_position[1])],
                         [int(last_average_position[0]), int(last_average_position[1])], (0, 255, 0), 5, cv2.LINE_AA)

            for ii in range(len(average_positions) - 1):
                cv2.line(background_copy, [int(average_positions[ii + 1][0]), int(average_positions[ii + 1][1])],
                         [int(average_positions[ii][0]), int(average_positions[ii][1])], (255, 0, 0), 5, cv2.LINE_AA)

            cv2.imshow('frame_flip', frame_flip)
            cv2.imshow('background', background_copy)

            k = cv2.waitKey(1)

            if k == ord('c'):
                average_positions.clear()
            if k == 27:
                exit()

            # T2 = time.time()

            # print(T2-T1)

        last_average_position = average_position
        average_position = get_point_without_outliers(point_list)
        if average_position[0] != 0 and average_position[1] != 0 and average_position[1] > w / 2:
            average_positions.append(average_position)
        



if __name__ == "__main__":
    start_camera()
