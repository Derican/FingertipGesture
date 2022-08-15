from Plot import *

zero_data = []
error_data = []
too_long_data = []

def checkPath(path):
    """
    description
    ---------
    Check average path
    
    param
    -------
    path: path containing matrices
    
    Returns
    -------
    "error" or "zero" or "too_long" or None
    
    """

    first_over_zero = False # the first time pressure over 10
    first_down_zero = False # the first time pressure down 10 after first_over_zero
    if len(path) > 3000:
        return "too_long"
    for frame in path:  # 一次采集
        sum_force = 0
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                sum_force += frame[y_coordinate][x_coordinate]
        if sum_force > 100:  # threshold = 10
            first_over_zero = True
            if first_down_zero:
                # over and down and over, error!!
                return "error"
        else:
            if first_over_zero:
                first_down_zero = True
    if not first_over_zero:
        # pressure never over 10, error!!
        return "zero"

def cleanAll():
    # pass
    for num_d in ["4", "6", "8", "10", "12"]:
        angles = DIRECTIONS_MAP[num_d]
        angles = np.concatenate((angles, [angles[0]]))
        for dir in os.listdir(BASE_DIR):
            print(dir)
            for i, c in enumerate(DIRECTIONS_MAP[num_d]):
                for j in range(5):
                    filename = os.path.join(BASE_DIR, dir, '0', str(num_d),
                                    str(i) + '_' + str(j) + '.npy')
                    path = np.load(filename)
                    check_path_val = checkPath(path)
                    if check_path_val == "zero":
                        zero_data.append(filename)
                    elif check_path_val == "error":
                        error_data.append(filename)
                    elif check_path_val == "too_long":
                        too_long_data.append(filename)
        print("{} done".format(num_d))
    np.save(os.path.join("new_data", "zero.npy"), zero_data)
    np.save(os.path.join("new_data", "error.npy"), error_data)
    np.save(os.path.join("new_data", "too_long.npy"), too_long_data)


if __name__ == "__main__":
    cleanAll()