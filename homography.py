from contour_solution import use_contour_to_get_position
from common import get_scree_wh, get_distance_point
import cv2
import numpy as np
from geometry import get_point_without_outliers

# 单应性矩阵相关的模块

# 使用单应性矩阵转换点
def use_Homography_transform_point(get_point, H):
    point = np.array([[get_point[0]], [get_point[1]], [1]])
    result = np.dot(H, point)
    result = result * (1.0 / result[2][0])

    return result[0][0], result[1][0]


# 得到单应性矩阵
def get_Homography(camera):
    cv2.namedWindow('Homography_win', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Homography_win', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    h, w = get_scree_wh()

    Homography_win = np.ones((w, h, 3), dtype=np.float32)

    point_num = 7

    points_window = np.array(
        [[h / 2, w / 7 * 2], [h / 13 * 5, w / 4 * 2], [h / 13 * 8, w / 4 * 2], [h / 3, w / 5 * 4],
         [h / 3 * 2, w / 5 * 4], [h / 2, w / 7 * 4], [h / 2, w / 7 * 6]],
        dtype=np.int32)

    points_mirror = []

    for point in points_window:
        cv2.circle(Homography_win, point, 5, (0, 255, 0), -1, cv2.LINE_AA)

    cv2.putText(Homography_win, 'Please touch the calibration points with your finger and press space',
                (int(h / 3), int(w / 10 * 9)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                2)

    index = 0
    # 需要校准的次数
    calibration_num = 3
    calibration_point_list = []
    cv2.circle(Homography_win, points_window[index], 10, (0, 0, 255), -1, cv2.LINE_AA)

    while True:
        ret, frame = camera.read()
        frame_flip = cv2.flip(frame, 1)
        get_position, is_get_position = use_contour_to_get_position(frame_flip)
        new_Homography_win = Homography_win.copy()
        # 不加copy的话，实际上是一个引用
        position = None
        if is_get_position:
            position = [get_position[0], get_position[1]]
            # position[1] = (position[1]-150) * 10
            print(position)
            # 转字符串
            cv2.putText(new_Homography_win, str(position), (int(h / 10), int(w / 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        cv2.imshow('Homography_win', new_Homography_win)

        # 空格，按下空格键存一次点的位置
        if cv2.waitKey(1) == 32 and is_get_position and index < point_num:
            calibration_point_list.append(position)
            cv2.circle(Homography_win, points_window[index], 10 + 5 * len(calibration_point_list), (255, 0, 0), 2,
                       cv2.LINE_AA)
            if len(calibration_point_list) == calibration_num:
                index = index + 1
                point_x = 0
                point_y = 0

                min_distance = 1000000
                for i in range(calibration_num):
                    for j in range(i + 1, calibration_num):
                        distance = get_distance_point(calibration_point_list[i], calibration_point_list[j])
                        if distance < min_distance:
                            min_distance = distance
                            point_x = (calibration_point_list[i][0] + calibration_point_list[j][0]) / 2
                            point_y = (calibration_point_list[i][1] + calibration_point_list[j][1]) / 2

                # [point_x, point_y] = get_point_without_outliers(calibration_point_list)

                points_mirror.append([point_x, point_y])

                calibration_point_list.clear()
                if index != point_num:
                    cv2.circle(Homography_win, points_window[index], 10, (0, 0, 255), -1, cv2.LINE_AA)
                else:
                    points_mirror = np.array([points_mirror], dtype=np.int32)
                    Homography, status = cv2.findHomography(points_mirror, points_window)
                    cv2.destroyWindow('Homography_win')
                    return Homography
