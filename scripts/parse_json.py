import json
import numpy as np

def parse_json(filepath):
    
    # Store the various camera parameters
    parameters = []
    
    # Read the JSON File
    file = open(filepath)
    
    # Fetch the JSON Object as a dictionary
    data = json.load(file)
    
    cx = data["cx"]["val"]
    cy = data["cy"]["val"]
    f = data["f"]["val"]
    ok1 = data["ok1"]["val"]
    ok2 = data["ok2"]["val"]
    ok3 = data["ok3"]["val"]
    ok4 = data["ok4"]["val"]
    ok5 = data["ok5"]["val"]
    ok6 = data["ok6"]["val"]
    op1 = data["op1"]["val"]
    op2 = data["op2"]["val"]
    
    camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    distortion_coefficients = np.array([ok1, ok2, op1, op2, ok3, ok4, ok5, ok6], dtype=np.float64)
    
    parameters.append([camera_matrix, distortion_coefficients])
    
    return parameters
    
    