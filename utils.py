from constants import *


def ACosAngleDist(ang1, ang2):
    return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                np.linalg.norm(ang2))


def getAveragePath(path,
                   align_to_first=True,
                   integer=False,
                   truncate=True,
                   use_threshold=100):
    """
    description
    ---------
    Get average path from path containing matrices to x, y, depth array respectively
    
    param
    -------
    path: path containing matrices
    align_to_first: whether output begins with (0,0)
    integer: whether output coordinates are integers
    
    Returns
    -------
    (x, y, depth)
    
    """

    points_x = []
    points_y = []
    depths = []
    points_x_cache = []
    points_y_cache = []
    depths_cache = []
    threshold = use_threshold  # use const
    first_over_threshold = False

    for frame in path:  # 一次采集
        sum_force = 0
        x_average = 0
        y_average = 0
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                sum_force += frame[y_coordinate][x_coordinate]
        if sum_force > 2 * threshold:  # if(sum_force > 0)  有效采集的阈值
            first_over_threshold = True
            if points_x_cache:
                points_x += points_x_cache
                points_y += points_y_cache
                depths += depths_cache
                points_x_cache.clear()
                points_y_cache.clear()
                depths_cache.clear()
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate  # 取平均值作为该压力点坐标
                    y_average += rate * y_coordinate
            if integer:
                points_x.append(int(x_average * 10))
                points_y.append(int((35 - y_average) * 10))
            else:
                points_x.append(x_average)
                points_y.append(35 - y_average)
            depths.append(sum_force)
        elif first_over_threshold and sum_force > 0:
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate  # 取平均值作为该压力点坐标
                    y_average += rate * y_coordinate
            if integer:
                points_x_cache.append(int(x_average * 10))
                points_y_cache.append(int((35 - y_average) * 10))
            else:
                points_x_cache.append(x_average)
                points_y_cache.append(35 - y_average)
            depths_cache.append(sum_force)

    if len(points_x) <= 0:
        return np.array(points_x), np.array(points_y), np.array(depths)
    if align_to_first:
        center_x = points_x[0]  # 起始点为(0, 0)，其他的都要减掉
        center_y = points_y[0]
        for i in range(len(points_x)):
            points_x[i] -= center_x
        for i in range(len(points_y)):
            points_y[i] -= center_y
    return np.array(points_x), np.array(points_y), np.array(depths)


def getSingleDirectionConfidenceList(v, num_d, person, mix=False):
    gauss_dict = {}

    if mix:
        config_filename = 'gauss_direction_mix.json'
    else:
        config_filename = 'gauss_direction_{num_d}.json'.format(num_d=num_d)
    # else:
    #     config_filename = 'config/gauss_direction_{num_d}_{name}.json'.format(
    #         num_d=num_d, name=person)
    with open(config_filename, 'r') as file:
        gauss_dict = json.load(file)

    if person is not None:
        with open("offset.json", "r") as offset_f:
            offset_dict = json.load(offset_f)
            if person in offset_dict:
                for (key, value) in gauss_dict.items():
                    gauss_dict[key][1] -= offset_dict[person]

    ang = np.arctan2(v[1], v[0])
    confidence_list = []
    for ix in range(num_d):
        val = np.exp(
            -((ang - gauss_dict[str(ix)][1]) / gauss_dict[str(ix)][2])**2 / 2)
        val_adj_up = np.exp(-((ang - 2 * np.pi - gauss_dict[str(ix)][1]) /
                              gauss_dict[str(ix)][2])**2 / 2)
        val_adj_dn = np.exp(-((ang + 2 * np.pi - gauss_dict[str(ix)][1]) /
                              gauss_dict[str(ix)][2])**2 / 2)
        # val = 1 - abs(ang - gauss_dict[str(ix)][1])
        # val_adj_up = 1 - abs(ang - 2 * np.pi - gauss_dict[str(ix)][1])
        # val_adj_dn = 1 - abs(ang + 2 * np.pi - gauss_dict[str(ix)][1])
        confidence_list.append(
            (gauss_dict[str(ix)][0], max(val, val_adj_up, val_adj_dn)))
    return sorted(confidence_list, key=lambda t: t[1], reverse=True)


def angleDistLinear(ang1, ang2):
    return min(abs(ang1 - ang2), abs(ang1 + 2 * np.pi - ang2),
               abs(ang1 - 2 * np.pi - ang2))


def angleDiffLinear(u, v):
    ang1 = np.arctan2(u[1], u[0])
    ang2 = np.arctan2(v[1], v[0])
    return 2 * np.pi - abs(ang1 - ang2) if abs(ang1 -
                                               ang2) >= np.pi else abs(ang1 -
                                                                       ang2)


def getDirections6(path, person, truncate=True, debug=False):

    NUMBER_OF_POINTS_THRES = 15
    LENGTH_THRES = 1.5
    R_THRES = 0.89
    R_SQUARED_THRES = 0.9
    ANGLE_THRESHOLD = np.pi / 4

    x, y, d = getAveragePath(path, truncate=truncate, use_threshold=0)
    x = gaussian_filter1d(x, sigma=8)
    y = gaussian_filter1d(y, sigma=8)
    ext_directions = [
        getSingleDirectionConfidenceList((x[_] - x[_ - 1], y[_] - y[_ - 1]), 6,
                                         person)[0][0]
        for _ in range(1, len(x))
    ]
    collected = []
    collected_corners = []
    i = 0
    while (i < len(ext_directions)):
        current_direction = ext_directions[i]
        j = i
        while (j < len(ext_directions)
               and ext_directions[j] == current_direction):
            j += 1
        k = j
        while (k < len(ext_directions)):
            if (angleDistLinear(DIRECTIONS_MAP['6'][current_direction],
                                DIRECTIONS_MAP['6'][ext_directions[k]]) >
                    np.pi / 2):
                break
            while (k < len(ext_directions)
                   and ext_directions[k] == ext_directions[j]):
                k += 1
            # slope, intercept, r_value, p_value, std_err = stats.linregress(
            #     x[i:k], y[i:k])
            # if (r_value**2 < R_SQUARED_THRES):
            #     break
            # current_direction = getDirectionFromSlope(slope, y[i], y[k])
            angle_diff = angleDiffLinear((x[j] - x[i], y[j] - y[i]),
                                         (x[k] - x[j], y[k] - y[j]))
            if angle_diff > ANGLE_THRESHOLD:
                break
            current_direction = getSingleDirectionConfidenceList(
                (x[k] - x[i], y[k] - y[i]), 6, person)[0][0]
            if debug:
                print("merge from ", i, " to ", k, " angle diff = ",
                      angle_diff)
            j = k
        if j - i >= NUMBER_OF_POINTS_THRES and np.sqrt(
            (x[j] - x[i])**2 + (y[j] - y[i])**2) > LENGTH_THRES:
            if (len(collected) > 0 and collected[-1] == current_direction):
                last_i, last_j = collected_corners[-1]
                collected_corners[-1] = (last_i, j)
            else:
                collected.append(current_direction)
                collected_corners.append((i, j))
        i = j

    if debug:
        print("collected: ", collected, collected_corners)

    return collected, collected_corners


def angleDistMatrix(ang1, ang2):
    return CONFUSION_MATRIX[ang1][ang2]


class PersonalGaussianDist:

    def __init__(self, person) -> None:
        offset_dict = json.load(open('offset.json', 'r'))
        if person in offset_dict:
            self.offset = offset_dict[person]
        else:
            self.offset = 0

    def dist(self, u, v):
        ang1 = np.arctan2(u[1], u[0]) + self.offset
        ang2 = np.arctan2(v[1], v[0])
        val = np.exp(-((ang1 - ang2) / ang2)**2 / 2)
        val_adj_up = np.exp(-((ang1 - 2 * np.pi - ang2) / ang2)**2 / 2)
        val_adj_dn = np.exp(-((ang1 + 2 * np.pi - ang2) / ang2)**2 / 2)
        return 1 - max(val, val_adj_up, val_adj_dn)


def getDirectionFromSlope(slope, y_start, y_end, person):
    if (y_start >= y_end):
        if (slope >= 0):
            return getSingleDirectionConfidenceList((-1, -slope), 6,
                                                    person)[0][0]
        else:
            return getSingleDirectionConfidenceList((1, slope), 6,
                                                    person)[0][0]
    else:
        if (slope >= 0):
            return getSingleDirectionConfidenceList((1, slope), 6,
                                                    person)[0][0]
        else:
            return getSingleDirectionConfidenceList((-1, -slope), 6,
                                                    person)[0][0]


def predict(path, person, target, debug=False):
    # get average path
    points_x = []
    points_y = []
    depths = []
    threshold = 200  # use const

    points_x = path[0][path[2] > threshold]
    points_y = path[1][path[2] > threshold]
    depths = path[2][path[2] > threshold]

    # pressure extrema based filter - phase I
    if len(depths) <= 0:
        return None
    clamped_d = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
    pressure_persistence_pairs = sorted(
        [t for t in RunPersistence(clamped_d) if t[1] > 0.05],
        key=lambda x: x[0])
    if len(pressure_persistence_pairs) <= 2:
        i = 1
        while i < len(depths) and depths[i] > depths[i - 1]:
            i += 1
        first_extrema = depths[i]
        while i >= 0 and depths[i] >= first_extrema * 0.3:
            i -= 1
        trunc_at_start = i
        i = len(depths) - 1
        while i > 1 and depths[i - 1] > depths[i]:
            i -= 1
        last_extrema = depths[i]
        while i < len(depths) and depths[i] >= last_extrema * 0.7:
            i += 1
        trunc_at_end = i
    elif len(pressure_persistence_pairs) == 3:
        smallest_extrema = depths[pressure_persistence_pairs[1][0]]
        i = 1
        while i < len(depths) and depths[i] < smallest_extrema * 0.3:
            i += 1
        trunc_at_start = i
        i = len(depths) - 1
        while i > trunc_at_start and depths[i] < smallest_extrema * 0.3:
            i -= 1
        trunc_at_end = i
    else:
        ppp = pressure_persistence_pairs[1:-1]
        smallest_extrema = np.min([depths[_[0]] for _ in ppp])
        i = 1
        while i < len(depths) and depths[i] < smallest_extrema * 0.3:
            i += 1
        trunc_at_start = i
        i = len(depths) - 1
        while i > trunc_at_start and depths[i] < smallest_extrema * 0.6:
            i -= 1
        trunc_at_end = i

    # get directions of 6
    NUMBER_OF_POINTS_THRES = [16, 9, 9, 16, 9, 9]
    LENGTH_THRES = [1.6, 1.5, 1.5, 1.6, 1.5, 1.5]
    ANGLE_THRESHOLD = np.pi / 4

    raw_x, raw_y, raw_d = points_x[trunc_at_start:trunc_at_end], points_y[
        trunc_at_start:trunc_at_end], depths[trunc_at_start:trunc_at_end]
    x = gaussian_filter1d(raw_x, sigma=8)
    y = gaussian_filter1d(raw_y, sigma=8)
    ext_directions = [
        getSingleDirectionConfidenceList((x[_] - x[_ - 1], y[_] - y[_ - 1]), 6,
                                         person)[0][0]
        for _ in range(1, len(x))
    ]
    collected = []
    collected_corners = []
    i = 0
    while (i < len(ext_directions)):
        current_direction = ext_directions[i]
        j = i
        while (j < len(ext_directions)
               and ext_directions[j] == current_direction):
            j += 1
        k = j
        while (k < len(ext_directions)):
            if (angleDistLinear(DIRECTIONS_MAP['6'][current_direction],
                                DIRECTIONS_MAP['6'][ext_directions[k]]) >
                    np.pi / 2):
                break
            while (k < len(ext_directions)
                   and ext_directions[k] == ext_directions[j]):
                k += 1
            # slope, intercept, r_value, p_value, std_err = stats.linregress(
            #     x[i:k], y[i:k])
            # if (r_value**2 < R_SQUARED_THRES):
            #     break
            # current_direction = getDirectionFromSlope(slope, y[i], y[k])
            angle_diff = angleDiffLinear((x[j] - x[i], y[j] - y[i]),
                                         (x[k] - x[j], y[k] - y[j]))
            if angle_diff > ANGLE_THRESHOLD:
                break
            current_direction = getSingleDirectionConfidenceList(
                (x[k] - x[i], y[k] - y[i]), 6, person)[0][0]
            if debug:
                print("merge from ", i, " to ", k, " angle diff = ",
                      angle_diff)
            j = k
        if j - i >= NUMBER_OF_POINTS_THRES[current_direction] and np.sqrt(
            (x[j] - x[i])**2 +
            (y[j] - y[i])**2) > LENGTH_THRES[current_direction]:
            if (len(collected) > 0 and collected[-1] == current_direction):
                last_i, last_j = collected_corners[-1]
                collected_corners[-1] = (last_i, j)
            else:
                collected.append(current_direction)
                collected_corners.append((i, j))
        i = j

    if debug:
        print("collected: ", collected, collected_corners)

    if len(collected) <= 0:
        return None

    # pressure extrema based filter - phase II
    END_LENGTH_THRES = (0.4, 0.4)

    if (len(collected_corners) >= 2):
        if (np.sqrt(
            (x[collected_corners[0][0]] - x[collected_corners[0][1]])**2 +
            (y[collected_corners[0][0]] -
             y[collected_corners[0][1]])**2) / np.sqrt(
                 (x[collected_corners[1][0]] - x[collected_corners[1][1]])**2 +
                 (y[collected_corners[1][0]] - y[collected_corners[1][1]])**2)
                < END_LENGTH_THRES[0]):
            if debug:
                print(
                    "ignore start, ratio: ",
                    np.sqrt((x[collected_corners[0][0]] -
                             x[collected_corners[0][1]])**2 +
                            (y[collected_corners[0][0]] -
                             y[collected_corners[0][1]])**2) /
                    np.sqrt((x[collected_corners[1][0]] -
                             x[collected_corners[1][1]])**2 +
                            (y[collected_corners[1][0]] -
                             y[collected_corners[1][1]])**2))
            collected.pop(0)
            collected_corners.pop(0)

    if (len(collected_corners) >= 2):
        if (np.sqrt(
            (x[collected_corners[-1][0]] - x[collected_corners[-1][1]])**2 +
            (y[collected_corners[-1][0]] - y[collected_corners[-1][1]])**2
        ) / np.sqrt(
            (x[collected_corners[-2][0]] - x[collected_corners[-2][1]])**2 +
            (y[collected_corners[-2][0]] - y[collected_corners[-2][1]])**2) <
                END_LENGTH_THRES[1]):
            if debug:
                print(
                    "ignore end, ratio: ",
                    np.sqrt((x[collected_corners[-1][0]] -
                             x[collected_corners[-1][1]])**2 +
                            (y[collected_corners[-1][0]] -
                             y[collected_corners[-1][1]])**2) /
                    np.sqrt((x[collected_corners[-2][0]] -
                             x[collected_corners[-2][1]])**2 +
                            (y[collected_corners[-2][0]] -
                             y[collected_corners[-2][1]])**2))
            collected.pop(-1)
            collected_corners.pop(-1)

    if (abs(len(collected) - len(DIRECTION_PATTERN6[target])) >= 3):
        return None

    # dynamic time warping pairing
    candi_q = []
    for ch in LETTER:
        std_directions = DIRECTION_PATTERN6[ch]
        d, cost_matrix, acc_cost_matrix, warping_path = dtw(
            collected,
            std_directions,
            dist=angleDistMatrix,
            w=abs(len(collected) - len(std_directions)),
            s=2)
        candi_q.append((d, ch))
    candi_q.sort()
    # min_dis = candi_q[0][0]
    # ans = []
    # for cand in candi_q:
    #     if cand[0] == min_dis:
    #         ans.append(cand[1])
    #     else:
    #         break
    # if target in ans:
    #     return True
    # if len(ans) > 1:
    #     raw_directions = [
    #         np.array([x[t] - x[s], y[t] - y[s]]) for s, t in collected_corners
    #     ]
    #     candi_q = []
    #     for ch in ans:
    #         std_directions = [
    #             np.array([
    #                 np.cos(DIRECTION_GAUSS_MODEL[str(_)][0]),
    #                 np.sin(DIRECTION_GAUSS_MODEL[str(_)][0])
    #             ]) for _ in DIRECTION_PATTERN6[ch]
    #         ]
    #         d, cost_matrix, acc_cost_matrix, warping_path = dtw(
    #             raw_directions,
    #             std_directions,
    #             dist=PersonalGaussianDist(person).dist,
    #             w=abs(len(collected) - len(std_directions)),
    #             s=2)
    #         candi_q.append((d, ch))
    #     candi_q.sort()
    pred = candi_q[0][1]
    if target == pred:
        return True
    else:
        if debug:
            print(target, candi_q[:min(5, len(candi_q))])
            plt.axis('scaled')
            plt.xlim(10, 20)
            plt.ylim(15, 25)

            plt.scatter(raw_x, raw_y, c='grey')

            # for _ in range(0, trunc_at_end - 1):
            #     plt.scatter([x[_ + 1]], [y[_ + 1]], c=COLORS[ext_directions[_]])
            for s, t in collected_corners:
                plt.scatter([x[s]], [y[s]], c='red')
                plt.scatter([x[t]], [y[t]], c='red')
                plt.text(x[s], y[s], str(s))
                plt.text(x[t], y[t], str(t))
            plt.show()
        return False


def saveFramesAsPath(frames, path):
    points_x = []
    points_y = []
    depths = []
    threshold = 0

    for frame in frames:
        sum_force = 0
        x_average = 0
        y_average = 0
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                sum_force += frame[y_coordinate][x_coordinate]
        if sum_force > threshold:
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate
                    y_average += rate * y_coordinate
            points_x.append(x_average)
            points_y.append(35 - y_average)
            depths.append(sum_force)
    np.save(path, np.array([points_x, points_y, depths]))
