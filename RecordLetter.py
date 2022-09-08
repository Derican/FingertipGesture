import threading
import sys, time, os
import traceback

from numpy.lib.npyio import save

sys.path.append('sensel-lib-wrappers/sensel-lib-python')
import sensel
import numpy as np
import matplotlib.pyplot as plt

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QGridLayout, QLabel, QMainWindow, QApplication, QWidget
from PyQt5.QtGui import QColor, QPalette, QBrush, QPixmap, QPainter, QRgba64
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from colour import Color
from pgcolorbar.colorlegend import ColorLegendItem

import argparse, socket
import random, itertools

HEIGHT = 105
WIDTH = 185
LEFT_BOUND = 79
UP_BOUND = 34
RIGHT_BOUND = 106
DOWN_BOUND = 68
MAX_LEN = 200

recording = False
interrupted = False
plotting = False

left_bound = 184
up_bound = 104
right_bound = 0
down_bound = 0

sum_frame = np.zeros((HEIGHT, WIDTH))
frame_each = np.zeros(
    (DOWN_BOUND + 1 - UP_BOUND, RIGHT_BOUND + 1 - LEFT_BOUND))
frame_series = []

frame_operator = lambda x: None


def open_sensel():
    handle = None
    error, device_list = sensel.getDeviceList()
    if device_list.num_devices != 0:
        error, handle = sensel.openDeviceByID(device_list.devices[0].idx)
    return handle


def init_frame():
    error = sensel.setFrameContent(handle, sensel.FRAME_CONTENT_PRESSURE_MASK)
    error, frame = sensel.allocateFrameData(handle)
    error = sensel.startScanning(handle)
    return frame


def scan_frames(frame, info: sensel.SenselSensorInfo):
    while not interrupted:
        error = sensel.readSensor(handle)
        error, num_frames = sensel.getNumAvailableFrames(handle)
        for i in range(num_frames):
            error = sensel.getFrame(handle, frame)
            save_frame(frame, info)


def save_frame(frame, info: sensel.SenselSensorInfo):
    global frame_series
    rows = info.num_rows
    cols = info.num_cols
    fs = np.zeros((DOWN_BOUND + 1 - UP_BOUND, RIGHT_BOUND + 1 - LEFT_BOUND))
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            fs[i - UP_BOUND][j - LEFT_BOUND] += frame.force_array[i * cols + j]
    frame_series.append(fs)


def close_sensel(frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--index',
        default=-1,
        help='specify the index of directory where the record will be saved')
    parser.add_argument(
        '-n',
        '--name',
        default='test',
        help='specify the name of participant, default as "test"')
    args = parser.parse_args()

    if args.name == 'test':
        save_dir = "study2/test"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        save_dir = "study2/%s" % args.name
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    handle = open_sensel()
    if handle:
        error, info = sensel.getSensorInfo(handle)
        frame = init_frame()

        u = threading.Thread(target=scan_frames, args=(frame, info))
        u.setDaemon(True)
        u.start()

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('localhost', 34827))

        candidates = []
        for rpt in range(5):
            s_candidates = list(
                itertools.product([chr(y) for y in range(97, 123)], [rpt]))
            random.shuffle(s_candidates)
            candidates.extend(s_candidates)
        candidate_index = 0
        current_record_num = candidates[candidate_index][1]
        while True:
            if plotting:
                code = input(
                    'Press Enter to stop... Or \'p\' to rewrite the previous')
                recording = False
                plotting = False

                np.save(
                    save_dir + "/" + str(candidates[candidate_index][0]) +
                    "_" + str(current_record_num) + ".npy", frame_series)
                if code == 'q':
                    interrupted = True
                    break
                if code != 'p':
                    candidate_index += 1
                if candidate_index >= len(candidates):
                    interrupted = True
                    break

                frame_series = []
            else:
                target = candidates[candidate_index][0]
                s.sendto(repr(target).encode('gbk'), ('localhost', 34826))

                code = input('Press Enter to start record')
                if (code == 'q'):
                    interrupted = True
                    break
                current_record_num = candidates[candidate_index][1]
                recording = True
                plotting = True
            if (candidate_index >= len(candidates)):
                break
        print("Task Complete! Thank you~")
        close_sensel(frame)