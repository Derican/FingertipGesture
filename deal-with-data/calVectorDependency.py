import pandas as pd
import numpy as np
import os
import math
import json
import random
import matplotlib.pyplot as plt
from plot import gatherCorner, calFirstCorner, calSecondCorner, loadData, calculatePoints, calAngle
from cleanWords import cleanWords, lowerCase
from match import genPoints, linear_rectangle, genPattern
from draw import genKeyPoint
from statsmodels.formula.api import ols
from statsmodels.api import graphics
from statsmodels.stats.anova import anova_lm

BASE_DIR = "../data/alphabeta_data_"
PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]
SAVE_MP_DIR = "../data/match_points/"  # save match points
SAVE_VD_DIR = "../data/vector_dependency_points/"  # save vector_dependency_points
# save average vector_dependency_points
SAVE_AVE_VD_DIR = "../data/ave_vector_dependency_points/"
STD_KB_POS = {
    'q': np.array([-474, 105]),
    'w': np.array([-370, 105]),
    'e': np.array([-265, 105]),
    'r': np.array([-161, 105]),
    't': np.array([-52, 105]),
    'y': np.array([51, 105]),
    'u': np.array([156, 105]),
    'i': np.array([262, 105]),
    'o': np.array([367, 105]),
    'p': np.array([469, 105]),
    'a': np.array([-446, 0]),
    's': np.array([-340, 0]),
    'd': np.array([-235, 0]),
    'f': np.array([-131, 0]),
    'g': np.array([-28, 0]),
    'h': np.array([78, 0]),
    'j': np.array([184, 0]),
    'k': np.array([292, 0]),
    'l': np.array([398, 0]),
    'z': np.array([-400, -105]),
    'x': np.array([-293, -105]),
    'c': np.array([-187, -105]),
    'v': np.array([-82, -105]),
    'b': np.array([23, -105]),
    'n': np.array([127, -105]),
    'm': np.array([232, -105])
}
allPattern = cleanWords()


def calVectorPair(person, sentence, word):
    # calculate angle in (sentence, word)
    # return [[],[]] first is pattern angle and second is actual angle
    patterns = np.array(genPattern(sentence, word, False)).reshape(-1, 2)
    if (len(patterns) < 1):
        print("word shorter than three letters, return")
        return [[], []]
    key_points = genKeyPoint(person, sentence, word)
    # print(patterns)
    key_vectors = [(np.array(key_points[i + 1][0:2]) -
                    np.array(key_points[i][0:2])).tolist()
                   for i in range(len(key_points) - 1)]
    # print(key_vectors)
    return [patterns.tolist(), key_vectors]


def saveVectorDependency():
    for person in PERSON:
        for i in range(1, 82):
            for j in range(len(allPattern[i - 1])):
                if os.path.exists(BASE_DIR + person + "/" + str(i) + "_" +
                                  str(j) + ".npy"):
                    with open(
                            SAVE_VD_DIR + person + "_" + str(i) + "_" +
                            str(j) + ".txt", "w") as f:
                        f.write(
                            json.dumps(
                                np.array(calVectorPair(person, i,
                                                       j)).tolist()))
                else:
                    print("error, lost data ", person, i, j,
                          allPattern[i - 1][j])
                    break
            print("done", i)


def rect2polar(x, y):
    r = np.linalg.norm(np.array([x, y]))
    theta = np.arctan2(y, x)
    return r, theta


def drawVectorDependency(person):
    fig, axes = plt.subplots(1, 2)
    xs = []
    ys = []
    xt = []
    yt = []
    for i in range(1, 82):
        for j in range(len(allPattern[i - 1])):
            if os.path.exists(SAVE_VD_DIR + person + "_" + str(i) + "_" +
                              str(j) + ".txt"):
                with open(
                        SAVE_VD_DIR + person + "_" + str(i) + "_" + str(j) +
                        ".txt", "r") as f:
                    data = json.loads(f.read())
                    if (len(data[1]) > 0 and len(data[0]) == len(data[1])):
                        for d in data[0]:
                            xs.append(rect2polar(d[0], d[1])[0])
                            xt.append(rect2polar(d[0], d[1])[1])
                        for d in data[1]:
                            ys.append(rect2polar(d[0], d[1])[0])
                            yt.append(rect2polar(d[0], d[1])[1])
            else:
                print("error, lost data ", person, i, j, allPattern[i - 1][j])
                break
    axes[0].scatter(xs, ys)
    axes[0].axis("scaled")
    axes[1].scatter(xt, yt)
    axes[1].axis("scaled")
    plt.show()


def genCSV(person):
    xs = []
    ys = []
    for i in range(1, 82):
        for j in range(len(allPattern[i - 1])):
            if os.path.exists(SAVE_VD_DIR + person + "_" + str(i) + "_" +
                              str(j) + ".txt"):
                with open(
                        SAVE_VD_DIR + person + "_" + str(i) + "_" + str(j) +
                        ".txt", "r") as f:
                    data = json.loads(f.read())
                    if (len(data[1]) > 0 and len(data[0]) == len(data[1])):
                        for d in data[0]:
                            xs.append(rect2polar(d[0], d[1])[0])
                        for d in data[1]:
                            ys.append(rect2polar(d[0], d[1])[0])
                        # for k in range(len(data[0])):
                        #     if rect2polar(data[0][k][0],
                        #                   data[0][k][1])[1] > 0 and rect2polar(
                        #                       data[0][k][0],
                        #                       data[0][k][1])[1] < 3.14:
                        #         xs.append(
                        #             rect2polar(data[0][k][0],
                        #                        data[0][k][1])[1])
                        #         ys.append(
                        #             rect2polar(data[1][k][0],
                        #                        data[1][k][1])[1])
            else:
                print("error, lost data ", person, i, j, allPattern[i - 1][j])
                break
    with open(SAVE_VD_DIR + person + ".csv", "w") as f:
        f.write("x,y\n")
        for i in range(len(xs)):
            f.write(str(xs[i]) + "," + str(ys[i]) + "\n")


def anova(person):
    file = SAVE_VD_DIR + person + ".csv"
    data = pd.read_csv(file)
    print(data)
    formula = 'y ~ x'
    ols_results = ols(formula, data).fit()
    print(ols_results.summary())
    fig = plt.figure(figsize=(15, 8))
    fig = graphics.plot_regress_exog(ols_results, "x", fig=fig)
    plt.show()


def genAveCSV(person, x_interval):
    xs = []
    ys = []
    for i in range(1, 82):
        for j in range(len(allPattern[i - 1])):
            if os.path.exists(SAVE_VD_DIR + person + "_" + str(i) + "_" +
                              str(j) + ".txt"):
                with open(
                        SAVE_VD_DIR + person + "_" + str(i) + "_" + str(j) +
                        ".txt", "r") as f:
                    data = json.loads(f.read())
                    if (len(data[1]) > 0 and len(data[0]) == len(data[1])):
                        for d in data[0]:
                            xs.append(d)
                        for d in data[1]:
                            ys.append(d)
            else:
                print("error, lost data ", person, i, j, allPattern[i - 1][j])
                break
    points = []
    ave_points = []
    for i in range(len(xs)):
        points.append([math.floor(xs[i] / x_interval) * x_interval, ys[i]])
    points.sort(key=lambda x: x[0])
    p = 0
    while (p < len(points)):
        saved_y = [points[p][1]]
        saved_x = points[p][0]
        p += 1
        if (p == len(points)):
            ave_points.append([saved_x, np.mean(saved_y)])
        while (p < len(points) and points[p][0] == saved_x):
            saved_y.append(points[p][1])
            p += 1
        ave_points.append([saved_x, np.mean(saved_y)])
    print(ave_points)
    with open(SAVE_AVE_VD_DIR + person + ".csv", "w") as f:
        f.write("x,y\n")
        for i in range(len(ave_points)):
            f.write(str(ave_points[i][0]) + "," + str(ave_points[i][1]) + "\n")
    plt.scatter([i[0] for i in ave_points], [i[1] for i in ave_points])
    plt.show()


def anovaAverage(person):  # average in x axis
    file = SAVE_AVE_VD_DIR + person + ".csv"
    data = pd.read_csv(file)
    # print(data)
    formula = 'y ~ x'
    ols_results = ols(formula, data).fit()
    print(ols_results.summary())


if __name__ == "__main__":
    # drawAngleDependency("xq")
    # for person in PERSON:
    #     genCSV(person)
    # anova("xq")
    # genCSV("lyh")
    # anova("lyh")
    # saveVectorDependency()
    drawVectorDependency("tty")