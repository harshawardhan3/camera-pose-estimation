import math

def get_center(points):
    x_coords = [point[1] for point in points]
    y_coords = [point[2] for point in points]
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    return (center_x, center_y)

def angle(p1, p2):
    a = math.degrees(math.atan2(p2[2] - p1[1], p2[1] - p1[0]))
    a += 180
    if a > 360:
        a - 360
    return a