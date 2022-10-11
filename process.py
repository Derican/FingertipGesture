from constants import *
from utils import *
from worker import Worker


class IdentifyStartAndEnd(Worker):

    def __init__(self) -> None:
        super().__init__(valid_study2_data)
        self.zero_data_filenames = np.load(os.path.join(
            STUDY2_DIR, 'zero.npy'))
        self.overlap_data_filenames = np.load(
            os.path.join(STUDY2_DIR, 'overlap.npy'))

    def operator(self, person):
        pressure_first = []
        pressure_start = []
        pressure_last = []
        pressure_end = []
        speed_first = []
        speed_start = []
        speed_last = []
        speed_end = []
        end_ppp = []
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                path_name = os.path.join(STUDY2_DIR, person + '1',
                                         t_l + '_' + str(rep) + '.npy')
                if not os.path.exists(path_name):
                    path_name = os.path.join(STUDY2_DIR, person,
                                             t_l + '_' + str(rep) + '.npy')
                if path_name in self.zero_data_filenames or path_name in self.overlap_data_filenames:
                    continue
                path = np.load(path_name)

                x, y, d = getAveragePath(path, truncate=False, use_threshold=0)
                x = gaussian_filter1d(x, sigma=8)
                y = gaussian_filter1d(y, sigma=8)
                d = gaussian_filter1d(d, sigma=4)
                if len(d) <= 1:
                    continue
                delta_d = [abs(d[i] - d[i - 1]) for i in range(1, len(d))]
                clamped_d = (delta_d - np.min(delta_d)) / (np.max(delta_d) -
                                                           np.min(delta_d))
                pressure_persistence_pairs = sorted(
                    [t for t in RunPersistence(clamped_d) if t[1] > 0.05],
                    key=lambda x: x[0])
                trunc_at_end = pressure_persistence_pairs[-2][0]
                if len(x) <= 0:
                    continue
                redundant_directions, directions_index = getDirections6(
                    path, None, truncate=False)
                identified_directions_index = []

                if len(redundant_directions) <= 0:
                    continue

                path_directions = [
                    np.array([
                        np.cos(DIRECTIONS_MAP['6'][i]),
                        np.sin(DIRECTIONS_MAP['6'][i])
                    ]) for i in redundant_directions
                ]
                std_directions = [
                    np.array([
                        np.cos(DIRECTIONS_MAP['6'][i]),
                        np.sin(DIRECTIONS_MAP['6'][i])
                    ]) for i in DIRECTION_PATTERN6[t_l]
                ]
                dis, warping_paths = dtw_ndim.warping_paths(
                    path_directions, std_directions)
                paths = dtaidistance.dtw.best_path(warping_paths)

                idx = 0
                while idx < len(paths):
                    match_list = []
                    current_std_idx = paths[idx][1]
                    while idx < len(
                            paths) and paths[idx][1] == current_std_idx:
                        match_list.append(paths[idx][0])
                        idx += 1
                    identified_directions_index.append(
                        directions_index[match_list[np.argmin([
                            ACosAngleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                            for m_l in match_list
                        ])]])

                valid_start = identified_directions_index[0][0]
                valid_end = identified_directions_index[-1][1]

                # temp_pressure = []
                # temp_speed = []

                # for s, t in directions_index:
                #     if s >= valid_start:
                #         break
                #     for i in range(s, t):
                #         if i > 0:
                #             temp_pressure.append(abs(d[i] - d[i - 1]))
                #             temp_speed.append(
                #                 np.linalg.norm(
                #                     (x[i] - x[i - 1], y[i] - y[i - 1])))

                # if len(temp_pressure) > 0:
                #     pressure_start.append(np.max(temp_pressure))
                #     speed_start.append(np.max(temp_speed))
                # else:
                #     for i in range(identified_directions_index[0][0],
                #                    identified_directions_index[0][1]):
                #         temp_pressure.append(abs(d[i] - d[i - 1]))
                #         temp_speed.append(
                #             np.linalg.norm((x[i] - x[i - 1], y[i] - y[i - 1])))
                #     pressure_first.append(np.max(temp_pressure))
                #     speed_first.append(np.max(temp_speed))

                # temp_pressure = []
                # temp_speed = []

                # for s, t in reversed(directions_index):
                #     if t <= valid_end:
                #         break
                #     for i in range(s, t):
                #         if i > 0:
                #             temp_pressure.append(abs(d[i] - d[i - 1]))
                #             temp_speed.append(
                #                 np.linalg.norm(
                #                     (x[i] - x[i - 1], y[i] - y[i - 1])))

                # if len(temp_pressure) > 0:
                #     pressure_end.append(np.max(temp_pressure))
                #     speed_end.append(np.max(temp_speed))
                # else:
                #     for i in range(identified_directions_index[0][0],
                #                    identified_directions_index[0][1]):
                #         temp_pressure.append(abs(d[i] - d[i - 1]))
                #         temp_speed.append(
                #             np.linalg.norm((x[i] - x[i - 1], y[i] - y[i - 1])))
                #     pressure_last.append(np.max(temp_pressure))
                #     speed_last.append(np.max(temp_speed))

                # fig, axes = plt.subplots(1, 5)

                # axes[0].set_xlim(-10, 10)
                # axes[0].set_ylim(-10, 10)
                # axes[1].set_ylim(-0.1, 1.5)

                # axes[0].scatter(x, y, c='grey')
                # axes[0].scatter(x[:valid_start], y[:valid_start], c='red')
                # axes[0].scatter(x[valid_end:], y[valid_end:], c='blue')

                # for i in range(0, valid_start):
                #     axes[2].scatter([i], [d[i]], c='red')
                #     if i > 0:
                #         axes[1].scatter([i], [
                #             np.linalg.norm((x[i] - x[i - 1], y[i] - y[i - 1]))
                #         ],
                #                         c='red')
                #         axes[3].scatter([i], [abs(d[i] - d[i - 1])], c='red')
                # for i in range(valid_start, valid_end):
                #     axes[2].scatter([i], [d[i]], c='grey')
                #     if i > 0:
                #         axes[1].scatter([i], [
                #             np.linalg.norm((x[i] - x[i - 1], y[i] - y[i - 1]))
                #         ],
                #                         c='grey')
                #         axes[3].scatter([i], [abs(d[i] - d[i - 1])], c='grey')
                # for i in range(valid_end, len(x)):
                #     axes[2].scatter([i], [d[i]], c='blue')
                #     if i > 0:
                #         axes[1].scatter([i], [
                #             np.linalg.norm((x[i] - x[i - 1], y[i] - y[i - 1]))
                #         ],
                #                         c='blue')
                #         axes[3].scatter([i], [abs(d[i] - d[i - 1])], c='blue')
                # for i in range(1, trunc_at_end):
                #     axes[4].scatter([i], [abs(d[i] - d[i - 1])], c='grey')
                # for i in range(trunc_at_end, len(d)):
                #     axes[4].scatter([i], [abs(d[i] - d[i - 1])], c='green')

                for id, (idx, pers) in enumerate(pressure_persistence_pairs):
                    if idx >= valid_end:
                        end_ppp.append(id - len(pressure_persistence_pairs))
                # plt.show()

        return pressure_start, pressure_first, pressure_last, pressure_end, speed_start, speed_first, speed_last, speed_end, end_ppp


def identifyStartAndEnd():

    worker = IdentifyStartAndEnd()
    result = worker.run()
    pressure_first = []
    pressure_start = []
    pressure_last = []
    pressure_end = []
    speed_first = []
    speed_start = []
    speed_last = []
    speed_end = []
    end_ppp = []
    for p1, p2, p3, p4, s1, s2, s3, s4, e1 in result:
        for ps, ps_l in list(
                zip([p1, p2, p3, p4, s1, s2, s3, s4, e1], [
                    pressure_start, pressure_first, pressure_last,
                    pressure_end, speed_start, speed_first, speed_last,
                    speed_end, end_ppp
                ])):
            if len(ps) > 0:
                ps_l.append(np.average(ps))
    print(np.average(end_ppp), np.std(end_ppp))
    # fig, axes = plt.subplots(1, 2)
    # x = list(range(len(pressure_start)))
    # axes[0].scatter(x, pressure_start, label='pressure_start')
    # axes[0].scatter(x, pressure_first, label='pressure_first')
    # axes[0].scatter(x, pressure_last, label='pressure_last')
    # axes[0].scatter(x, pressure_end, label='pressure_end')
    # axes[0].legend()
    # axes[1].scatter(x, speed_start, label='speed_start')
    # axes[1].scatter(x, speed_first, label='speed_first')
    # axes[1].scatter(x, speed_last, label='speed_last')
    # axes[1].scatter(x, speed_end, label='speed_end')
    # axes[1].legend()
    # plt.show()
    fig, axes = plt.subplots(1, 2)
    df = pd.DataFrame({
        'pressure':
        pressure_start + pressure_first + pressure_last + pressure_end,
        'label1':
        ['start'] * len(pressure_start) + ['first'] * len(pressure_first) +
        ['last'] * len(pressure_last) + ['end'] * len(pressure_end),
    })
    sns.boxplot(x='label1', y='pressure', data=df, ax=axes[0])
    df = pd.DataFrame({
        'speed':
        speed_start + speed_first + speed_last + speed_end,
        'label2': ['start'] * len(speed_start) + ['first'] * len(speed_first) +
        ['last'] * len(speed_last) + ['end'] * len(speed_end),
    })
    sns.boxplot(x='label2', y='speed', data=df, ax=axes[1])
    plt.show()
    print(np.median(pressure_start))
    print(np.median(pressure_first))
    print(np.median(pressure_last))
    print(np.median(pressure_end))
    print(np.median(speed_start))
    print(np.median(speed_first))
    print(np.median(speed_last))
    print(np.median(speed_end))


class CalculateAccuracy(Worker):

    def __init__(self, target_set) -> None:
        super().__init__(target_set)
        self.zero_data_filenames = np.load(os.path.join(
            STUDY2_DIR, 'zero.npy'))
        self.overlap_data_filenames = np.load(
            os.path.join(STUDY2_DIR, 'overlap.npy'))

    def operator(self, person):
        total = 0
        correct = 0
        overlap = 0
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                path_name = os.path.join(STUDY2_DIR, person + '1',
                                         t_l + '_' + str(rep) + '.npy')
                personal = False
                if not os.path.exists(path_name):
                    path_name = os.path.join(STUDY2_DIR, person,
                                             t_l + '_' + str(rep) + '.npy')
                    personal = True
                path_name = os.path.join(STUDY2_DIR_ALTER, person,
                                         t_l + '_' + str(rep) + '.npy')
                path = np.load(path_name)

                total += 1
                ans = predict(path,
                              person if personal else None,
                              t_l,
                              debug=False)
                if ans:
                    correct += 1
                elif ans is None:
                    overlap += 1
        return np.array([total, correct, overlap])


def calculateAccuracy():
    worker = CalculateAccuracy(valid_study2_data)
    result = worker.run()
    print(np.sum(np.array(result), axis=0))


class CalibrateData(Worker):

    def __init__(self, target_set) -> None:
        super().__init__(target_set)
        self.offset_dict = json.load(open('offset.json', 'r'))

    def operator(self, person):
        if person not in self.offset_dict:
            return

        if not os.path.exists(os.path.join(STUDY2_DIR_ALTER, person)):
            os.mkdir(os.path.join(STUDY2_DIR_ALTER, person))

        offset = self.offset_dict[person]

        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                path_name = os.path.join(STUDY2_DIR, person + '1',
                                         t_l + '_' + str(rep) + '.npy')
                personal = False
                if not os.path.exists(path_name):
                    path_name = os.path.join(STUDY2_DIR, person,
                                             t_l + '_' + str(rep) + '.npy')
                    personal = True
                path = np.load(path_name)

                points_x = []
                points_y = []
                depths = []
                threshold = 0

                for frame in path:
                    sum_force = 0
                    x_average = 0
                    y_average = 0
                    for y_coordinate in range(len(frame)):
                        for x_coordinate in range(len(frame[y_coordinate])):
                            sum_force += frame[y_coordinate][x_coordinate]
                    if sum_force > threshold:
                        for y_coordinate in range(len(frame)):
                            for x_coordinate in range(len(
                                    frame[y_coordinate])):
                                rate = frame[y_coordinate][
                                    x_coordinate] / sum_force
                                x_average += rate * x_coordinate
                                y_average += rate * y_coordinate
                        points_x.append(x_average)
                        points_y.append(35 - y_average)
                        depths.append(sum_force)
                np.save(
                    os.path.join(STUDY2_DIR_ALTER, person,
                                 t_l + '_' + str(rep) + '.npy'),
                    np.array([points_x, points_y, depths]))


def calibrateData():
    worker = CalibrateData(valid_study2_data)
    worker.run()


if __name__ == '__main__':
    # identifyStartAndEnd()
    # calibrateData()
    calculateAccuracy()