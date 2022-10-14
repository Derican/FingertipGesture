import argparse, os
import numpy as np
from Plot import LETTER, getAveragePath, gaussian_filter1d, getDirections6, DIRECTION_PATTERN6, dtw, CONFUSION_MATRIX
from CleanData import checkPath, checkOverlap


def angleDist(ang1, ang2):
    return CONFUSION_MATRIX[ang1][ang2]


def checkError(path, target_ch):
    x, y, dep = getAveragePath(path)
    if (len(x) <= 0):
        return True
    x = gaussian_filter1d(x, sigma=8)
    y = gaussian_filter1d(y, sigma=8)

    candi_q = []
    path_directions = getDirections6(path, None)
    if (len(path_directions) <= 0):
        return True
    if (abs(len(path_directions) - len(DIRECTION_PATTERN6[target_ch])) >= 3):
        return True
    for ch in LETTER:
        std_directions = DIRECTION_PATTERN6[ch]
        d, cost_matrix, acc_cost_matrix, warping_path = dtw(
            path_directions,
            std_directions,
            dist=angleDist,
            w=abs(len(path_directions) - len(std_directions)),
            s=2)
        candi_q.append((d, ch))
    candi_q.sort()
    min_dis = candi_q[0][0]
    ans = []
    for cand in candi_q:
        if cand[0] == min_dis:
            ans.append(cand[1])
        else:
            break
    return target_ch not in ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--person',
                        help='specify the person you want to look into')
    args = parser.parse_args()

    TARGET_DIR = 'study2/' + args.person

    total = 0
    valid = 0

    for i, c in enumerate(LETTER):
        for j in range(5):
            filename = os.path.join(TARGET_DIR, c + '_' + str(j) + '.npy')
            if not os.path.exists(filename):
                continue
            path = np.load(filename)
            check_path_val = checkPath(path)
            total += 1
            if check_path_val == "zero":
                print("Zero: ", filename)
                continue
            if checkOverlap(path, None, c):
                print("Overlap: ", filename)
                continue
            if checkError(path, c):
                print("Error: ", filename)
                continue
            valid += 1

    print("Valid num: ", valid, " Total: ", total)
