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

    first_over_zero = False  # the first time pressure over 10
    first_down_zero = False  # the first time pressure down 10 after first_over_zero
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


def cleanAll2():
    for dir in os.listdir(STUDY2_DIR):
        if 'test' in dir or not os.path.isdir(os.path.join(STUDY2_DIR, dir)):
            continue
        for i, c in enumerate(LETTER):
            for j in range(5):
                filename = os.path.join(STUDY2_DIR, dir,
                                        c + '_' + str(j) + '.npy')
                path = np.load(filename)
                check_path_val = checkPath(path)
                if check_path_val == "zero":
                    zero_data.append(filename)
                elif check_path_val == "error":
                    error_data.append(filename)
                elif check_path_val == "too_long":
                    too_long_data.append(filename)
    np.save(os.path.join(STUDY2_DIR, "zero.npy"), zero_data)
    np.save(os.path.join(STUDY2_DIR, "error.npy"), error_data)
    np.save(os.path.join(STUDY2_DIR, "too_long.npy"), too_long_data)


def calOffset():
    offset_dict = {}
    for dir in os.listdir(STUDY2_DIR):
        if 'test' in dir or not os.path.isdir(os.path.join(STUDY2_DIR, dir)):
            continue
        c = 'l'
        down_set = []
        for j in range(5):
            filename = os.path.join(STUDY2_DIR, dir, c + '_' + str(j) + '.npy')
            path = np.load(filename)
            x, y, d = getAveragePath(path)
            x = gaussian_filter1d(x, sigma=8)
            y = gaussian_filter1d(y, sigma=8)
            directions_index, redundant_8directions, weights = getNumberOfDirections(
                path, 6, None)
            if (len(weights) <= 0):
                continue
            weights_sorted_index = np.argsort(weights)
            s, t = directions_index[0]
            ang = np.arctan2(y[t] - y[s], x[t] - x[s])
            if ang >= -1.8 and ang <= -1.4:
                down_set.append(ang)
            # plotOneLettersCorner8(path)
        # print(dir, down_set, -1.5926688431505287 - np.average(down_set))
        offset_dict[dir] = -1.5926688431505287 - np.average(down_set)
    with open('offset.json', 'w') as out:
        json.dump(offset_dict, out)


CONFUSION_MATRIX = [[0, 1, 5, 5, 5, 1], [1, 0, 2, 5, 5, 5], [5, 2, 0, 1, 5, 5],
                    [5, 5, 1, 0, 1, 5], [5, 5, 5, 1, 0, 2], [1, 5, 5, 5, 2, 0]]


def angleDist(ang1, ang2):
    return CONFUSION_MATRIX[ang1][ang2]


def checkOverlap(path, person, target_ch):
    path_directions = getDirections6(path, person)
    std_directions = [i for i in DIRECTION_PATTERN6[target_ch]]
    paths = dtaidistance.dtw.warping_path(path_directions, std_directions)

    overlap = False
    idx = 0
    while idx < len(paths):
        match_list = []
        current_std_idx = paths[idx][1]
        while idx < len(paths) and paths[idx][1] == current_std_idx:
            match_list.append(paths[idx][0])
            idx += 1
        min_dis = np.min([
            angleDist(path_directions[m_l], std_directions[current_std_idx])
            for m_l in match_list
        ])
        if min_dis > 2:
            overlap = True
            break

    return overlap


def calOverlap():

    error_data_filenames = np.load(os.path.join(STUDY2_DIR, 'error.npy'))
    too_long_data_filenames = np.load(os.path.join(STUDY2_DIR, 'too_long.npy'))

    overlap_data = []
    for dir in os.listdir(STUDY2_DIR):
        if dir == 'test' or not os.path.isdir(os.path.join(STUDY2_DIR, dir)):
            continue

        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path_name = os.path.join(STUDY2_DIR, dir,
                                             "%s_%d.npy" % (t_l, rep))
                    path = np.load(path_name)

                    path_directions = getDirections6(path, dir)
                    std_directions = [i for i in DIRECTION_PATTERN6[t_l]]
                    paths = dtaidistance.dtw.warping_path(
                        path_directions, std_directions)

                    overlap = False
                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        min_dis = np.min([
                            angleDist(path_directions[m_l],
                                      std_directions[current_std_idx])
                            for m_l in match_list
                        ])
                        if min_dis > 2:
                            overlap = True
                            break

                    if overlap:
                        overlap_data.append(path_name)
                        # plotDirections6(path, dir, t_l_i, rep)

                except Exception as e:
                    print(str(e))

    print(len(overlap_data))
    np.save("study2/overlap.npy", np.array(overlap_data))


if __name__ == "__main__":
    calOverlap()