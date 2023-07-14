import numpy as np
import math 

# Script to compute errors for each point compared to the ellipse

def point_to_ellipse_distance(point, ellipse):
    center, axes, angle = ellipse
    axes = axes[0] / 2, axes[1] / 2
    angle = np.deg2rad(angle)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    
    rotated_x = cos_angle * dx + sin_angle * dy
    rotated_y = -sin_angle * dx + cos_angle * dy
    
    normalized_x = rotated_x / axes[0]
    normalized_y = rotated_y / axes[1]
    
    distance = np.sqrt(normalized_x ** 2 + normalized_y ** 2)
    return abs(distance - 1)

def calculate_residuals(points, ellipse):
    (x0, y0), (a, b), angle = ellipse
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    residuals = []
    for point in points:
        x, y = point[1:3]
        x_diff = x - x0
        y_diff = y - y0
        x_transformed = cos_angle * x_diff + sin_angle * y_diff
        y_transformed = -sin_angle * x_diff + cos_angle * y_diff
        error = ((x_transformed / a) ** 2 + (y_transformed / b) ** 2) - 1
        residuals.append(error)
    return residuals
