import logging
from itertools import product
from math import atan2, sqrt
from multiprocessing import Pool
import numpy as np
import argparse
import os
import json
import re
import matplotlib.pyplot as plt
import dtaidistance
from dtaidistance import dtw_ndim
from dtw import dtw
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from Calculate import calculatePoints, genPointLabels
from persistence1d import RunPersistence
from queue import PriorityQueue
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from mpl_toolkits.mplot3d import axes3d
from scipy import stats
from matplotlib.animation import ArtistAnimation

STUDY1_DIR = 'study1'
STUDY2_DIR = 'study2'
STUDY2_DIR_ALTER = 'study2-alter'
STUDY3_DIR = 'study3'

valid_study2_data = [
    'cly', 'djy', 'gyf', 'jyw', 'lly', 'lm', 'stp', 'wsw', 'ytj', 'yxy', 'zxw',
    'zyw', 'jjx', 'hz', 'lyj', 'xrh'
]

LETTER = [chr(y) for y in range(97, 123)]
DIRECTIONS_MAP = {
    '4': [np.pi / 2, 0, -np.pi / 2, -np.pi],
    '6': [
        np.pi / 2, np.pi / 6, -np.pi / 6, -np.pi / 2, -5 * np.pi / 6,
        5 * np.pi / 6
    ],
    '8': [
        np.pi / 2, np.pi / 4, 0, -np.pi / 4, -np.pi / 2, -3 * np.pi / 4,
        -np.pi, 3 * np.pi / 4
    ],
    '10': [
        np.pi / 2, 3 * np.pi / 10, np.pi / 10, -np.pi / 10, -3 * np.pi / 10,
        -np.pi / 2, -7 * np.pi / 10, -9 * np.pi / 10, 9 * np.pi / 10,
        7 * np.pi / 10
    ],
    '12': [
        np.pi / 2, np.pi / 3, np.pi / 6, 0, -np.pi / 6, -np.pi / 3, -np.pi / 2,
        -2 * np.pi / 3, -5 * np.pi / 6, -np.pi, 5 * np.pi / 6, 2 * np.pi / 3
    ]
}
COLORS = [
    'red', 'chocolate', 'darkorange', 'yellow', 'lawngreen', 'green', 'cyan',
    'slategrey', 'blue', 'darkviolet', 'magenta', 'hotpink'
]
DIRECTION_PATTERN6 = {
    'a': np.array([4, 1, 3]),
    'b': np.array([3, 1, 3, 5]),
    'c': np.array([4, 1]),
    'd': np.array([4, 1, 0]),
    'e': np.array([1, 4, 3, 1]),
    'f': np.array([4, 3, 0, 1]),
    'g': np.array([5, 2, 3, 5]),
    'h': np.array([3, 1, 3]),
    'i': np.array([3, 0]),
    'j': np.array([3, 5]),
    'k': np.array([3, 1, 4, 2]),
    'l': np.array([3]),
    'm': np.array([3, 0, 3, 0, 3]),
    'n': np.array([3, 0, 3]),
    'o': np.array([4, 2, 0]),
    'p': np.array([3, 0, 2, 5]),
    'q': np.array([5, 3, 1, 3]),
    'r': np.array([3, 0, 2]),
    's': np.array([4, 1, 4]),
    't': np.array([2, 5, 3]),
    'u': np.array([3, 1, 0]),
    'v': np.array([2, 1]),
    'w': np.array([3, 0, 3, 0]),
    'x': np.array([2, 5, 4]),
    'y': np.array([2, 1, 4]),
    'z': np.array([1, 4, 1]),
}

# CONFUSION_MATRIX = [[0, 11, 15, 15, 15, 11], [11, 0, 12, 15, 15, 15],
#                     [15, 12, 0, 11, 15, 15], [15, 15, 11, 0, 11, 15],
#                     [15, 15, 15, 11, 0, 12], [11, 15, 15, 15, 12, 0]]

# CONFUSION_MATRIX = [[0, 1, 5, 5, 5, 1], [1, 0, 2, 5, 5, 5], [5, 2, 0, 1, 5, 5],
#                     [5, 5, 1, 0, 1, 5], [5, 5, 5, 1, 0, 2], [1, 5, 5, 5, 2, 0]]

CONFUSION_MATRIX = [
    [0., 2.2732943, 4.6507807, 4.07541655, 4.90209513, 2.77046783],
    [3.00296413, 0., 3.27995091, 4.73220324, 3.69611131, np.inf],
    [3.25809654, 3.91487607, 0., 3.78134468, 3.15273602, 4.60802325],
    [3.57939522, 3.29171315, 2.42671571, 0., 2.83218082, 4.73516592],
    [5.45425217, 3.17698488, 5.74193424, 4.35563988, 0., 3.89610755],
    [3.14988295, 5.75257264, 3.55534806, 2.75684037, 2.66153019, 0.]
]
DIRECTION_GAUSS_MODEL = {
    "0": [1.618184136895924, 0.11307039816195545],
    "1": [0.7771298861048253, 0.3365436034421946],
    "2": [-0.7267845993056391, 0.3681828074876641],
    "3": [-1.5926688431505287, 0.10247877797775981],
    "4": [-2.709753283162458, 0.3522567343647264],
    "5": [2.2995040214409124, 0.3145086534289635]
}