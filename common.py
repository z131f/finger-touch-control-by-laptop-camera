import math
import win32gui
import win32con
import win32print


# 一些通用方法

# 获取两个点之间的距离
def get_distance_point(Point0, PointA):
    # 两个点之间的位置
    distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow((Point0[1] - PointA[1]), 2)
    distance = math.sqrt(distance)
    return distance


# 获取屏幕的分辨率
def get_scree_wh():
    hDC = win32gui.GetDC(0)
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return h, w
