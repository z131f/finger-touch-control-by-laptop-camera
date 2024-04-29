from common import get_distance_point
import numpy as np
# 得到平均中心点
def get_centers_of_points(points):
    length = len(points)
    if length > 0:
        g_center = [sum([x[0] for x in points]) / length, sum([x[1] for x in points]) / length]
    else:
        g_center = [-1, -1]
    return g_center


# 删除离群点
def get_point_without_outliers(points):
    if not len(points) > 0:
        return [-1, -1]
    center = get_centers_of_points(points)
    distant = []
    for point in points:
        distant.append(get_distance_point(point, center))

    distant_std = np.std(distant)

    result_point = []

    for i in range(len(points)):

        zscore = distant[i] / distant_std
        if not zscore > 2.2:
            result_point.append(points[i])


    return get_centers_of_points(result_point)


