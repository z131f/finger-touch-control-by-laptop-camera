import cv2
import numpy as np
import math

# 利用opencv附带的轮廓方法来获取坐标信息


# 记录上一帧预测出的位置
last_position = [-1, 0]
# 上一个重合轮廓的信息：上一帧是否重合，上一帧坐标
pre_OverlappingContour_info = [False, [-1, -1]]


# 使用轮廓的方法获取点
def use_contour_to_get_position(frame_flip):
    binaryImage = get_BinaryImage(frame_flip)
    finger_real, finger_mirror, is_fingerReturn, is_onlyOne, is_water = get_finger(binaryImage)

    if is_fingerReturn:
        if is_onlyOne:
            cv2.drawContours(frame_flip, finger_real, -1, (255, 0, 0), 3)
        else:
            # 红色
            cv2.drawContours(frame_flip, finger_real, -1, (0, 255, 0), 3)
            # 绿色
            cv2.drawContours(frame_flip, finger_mirror, -1, (0, 0, 255), 3)

        cv2.imshow("Finger", frame_flip)

        position, is_get_position = calculate_contact_position(finger_real, finger_mirror, is_onlyOne, is_water)
        return position, is_get_position
    else:
        return None, False


# 两个点之间的位置
def get_distance_point(Point0, PointA):
    distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow((Point0[1] - PointA[1]), 2)
    distance = math.sqrt(distance)
    return distance


# 改一下代码，两种方法都采用


# 返回位置，是否接触
def calculate_contact_position(finger_real, finger_mirror, is_OnlyOne, is_water):
    # 标识为全局变量
    global last_position

    area = cv2.contourArea(finger_real)
    area2 = cv2.contourArea(finger_mirror)
    bottom_mirror = tuple(finger_mirror[finger_mirror[:, :, 1].argmin()][0])
    top_rela = tuple(finger_real[finger_real[:, :, 1].argmax()][0])

    global pre_OverlappingContour_info

    if ((area2 / area > 7) or (area / area2 > 7) or (is_OnlyOne and area > 6800)) and last_position[0] != -1:
        return last_position, False
    else:
        # pre_OverlappingContour_info[0] = False

        if is_water:
            print("www")
            ellipse_real = cv2.fitEllipse(finger_real)
            ellipse_mirror = cv2.fitEllipse(finger_mirror)

            mask1 = np.zeros((480, 640, 1), dtype=np.float32)

            cv2.ellipse(mask1, ellipse_real, (255, 255, 255))
            cv2.ellipse(mask1, ellipse_mirror, (255, 255, 255))

            cv2.imshow('mask1', mask1)

            ellipse_real = cv2.fitEllipse(finger_real)
            ellipse_mirror = cv2.fitEllipse(finger_mirror)

            # 三个参数分别是 圆心 长短轴 偏转角度

            mask1 = np.zeros((480, 640, 1), dtype=np.float32)
            cv2.ellipse(mask1, ellipse_real, (255, 255, 255), 1)

            mask2 = np.zeros((480, 640, 1), dtype=np.float32)
            cv2.ellipse(mask2, ellipse_mirror, (255, 255, 255), 1)

            mask = cv2.bitwise_and(mask1, mask2)

            points = np.where(mask == 255)

            if points[0].size == 2 and points[1].size == 2:
                position = (int((points[1][0] + points[1][1]) / 2), int((points[0][0] + points[0][1]) / 2))
                last_position = position
                return last_position, True
            else:
                last_position = [-1, -1]
                return last_position, False
        else:
            pre_OverlappingContour_info[0] = False
            last_position = [int((top_rela[0] + bottom_mirror[0]) / 2), int((top_rela[1] + bottom_mirror[1]) / 2)]
            if get_distance_point(top_rela, bottom_mirror) < 50:
                return last_position, True
            else:
                return last_position, False


# 获取手指轮廓
def get_finger(binaryImage, *frame_flip):
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    con_list = []
    for cnt in contours:
        center, size, angle = cv2.minAreaRect(cnt)
        if (35 < angle < 55) or (125 < angle < 145) or size[0] < 13 or size[1] < 25:
            continue
        area = cv2.contourArea(cnt)
        if area > 50000 or area < 300:
            continue
        con_list.append((cnt, area))
    if len(con_list) == 0:
        return None, None, False, False, False
    if len(con_list) >= 2:
        con_list.sort(key=lambda x: x[1])
        center1, size1, angle1 = cv2.minAreaRect(con_list[-1][0])
        center2, size2, angle2 = cv2.minAreaRect(con_list[-2][0])
        if center1[1] < center2[1]:
            return con_list[-1][0], con_list[-2][0], True, False, False
        else:
            return con_list[-2][0], con_list[-1][0], True, False, False
    else:
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binaryImage, kernel, iterations=3)
        dist_transfrom = cv2.distanceTransform(binaryImage, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transfrom, 0.7 * dist_transfrom.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        color_image = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color_image, markers)
        color_image[markers == -1] = (0, 0, 0)

        new_b = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        new_b = cv2.erode(new_b, kernel, iterations=2)

        cv2.imshow('markers', new_b)

        contours, hierarchy = cv2.findContours(new_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con2_list = []
        for cnt in contours:
            center, size, angle = cv2.minAreaRect(cnt)
            if (35 < angle < 55) or (125 < angle < 145) or size[0] < 13 or size[1] < 25:
                continue
            area = cv2.contourArea(cnt)
            if area > 50000 or area < 300:
                continue
            con2_list.append((cnt, area))
        if len(con2_list) >= 2:
            con2_list.sort(key=lambda x: x[1])
            center1, size1, angle1 = cv2.minAreaRect(con2_list[-1][0])
            center2, size2, angle2 = cv2.minAreaRect(con2_list[-2][0])
            if center1[1] < center2[1]:
                return con2_list[-1][0], con2_list[-2][0], True, False, True
            else:
                return con2_list[-2][0], con2_list[-1][0], True, False, True
        else:
            return None, None, False, False, False


# 获取灰度图
def get_BinaryImage(frame):
    frame2 = frame
    kernel = np.ones((3, 3), np.uint8)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2LAB)
    frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
    frame2 = cv2.inRange(frame2[:, :, 2], np.array([70]), np.array([120]))

    frame2 = cv2.erode(frame2, kernel, iterations=4)
    frame2 = cv2.dilate(frame2, kernel, iterations=2)
    frame2 = cv2.erode(frame2, kernel, iterations=2)
    frame2 = cv2.dilate(frame2, kernel, iterations=2)

    # 开运算，先腐蚀后膨胀，效果不好
    # frame2 = cv2.morphologyEx(frame2, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=10)
    cv2.imshow('frame', frame2)
    return frame2
