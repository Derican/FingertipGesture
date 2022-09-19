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

import argparse

candidates = [chr(y) for y in range(97, 123)]

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
candidate_index = 0
current_record_num = 0

left_bound = 184
up_bound = 104
right_bound = 0
down_bound = 0

sum_frame = np.zeros((HEIGHT, WIDTH))
frame_each = np.zeros(
    (DOWN_BOUND + 1 - UP_BOUND, RIGHT_BOUND + 1 - LEFT_BOUND))
frame_series = []

frame_operator = lambda x: None


class RealtimePlot(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        black, red = Color('black'), Color('red')
        colors = black.range_to(red, 256)
        colors_array = np.array(
            [np.array(color.get_rgb()) * 255 for color in colors])
        look_up_table = colors_array.astype(np.uint8)

        self.img = pg.ImageItem()
        self.img.setLookupTable(look_up_table)
        self.view_box = pg.ViewBox()
        self.view_box.addItem(self.img)
        self.plot = pg.PlotItem(viewBox=self.view_box)
        self.addItem(self.plot)

        self.color_bar = ColorLegendItem(imageItem=self.img,
                                         showHistogram=True)
        self.addItem(self.color_bar)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)  # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.setData)

    def setData(self):
        self.img.setImage(frame_each / 50)


class FeedbackRecordWindow(QWidget):

    def __init__(self, parent=None):
        super(FeedbackRecordWindow, self).__init__(parent)
        self.setWindowTitle("FeedbackRecordWindow")

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)  # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.repaint)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        qp = QPainter()
        qp.begin(self)
        qb = QBrush(Qt.BrushStyle.SolidPattern)
        pixmap = QPixmap("./keyboard.png")
        qp.drawPixmap(100, 80, 360, 440, pixmap)
        for i in range(35):
            for j in range(28):
                red = (frame_each[i][j] /
                       25) if frame_each[i][j] / 25 <= 1 else 1
                clr = QColor(int(red * 255), 0, 0, int(red * 100))
                qb.setColor(clr)
                qp.setBrush(qb)
                qp.drawRect(j * 20, i * 20, 20, 20)
        qp.end()


def wait_for_enter():
    global recording, interrupted, plotting, candidate_index, current_record_num
    if args.interactive:
        input("Press Enter to stop...")
        interrupted = True
    else:
        while True:
            try:
                if (recording):
                    code = input(
                        'Press Enter to stop... Or \'p\' to return to previous'
                    )
                    recording = False
                    plotting = True
                    if (code == 'q'):
                        interrupted = True
                        break
                    if code != 'p':
                        current_record_num += 1
                    if current_record_num >= 5:
                        current_record_num = 0
                        candidate_index += 1
                    if candidate_index >= 26:
                        interrupted = True
                        break
                else:
                    code = input('Press Enter to start record of ' +
                                 candidates[candidate_index])
                    if (code == 'q'):
                        interrupted = True
                        if args.feedback:
                            QtCore.QCoreApplication.instance().quit()
                        break
                    recording = True

            except:
                interrupted = True
                if args.feedback:
                    QtCore.QCoreApplication.instance().quit()
                break
    return


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
            if recording:
                frame_operator(frame, info)
            if args.feedback:
                print_each_frame(frame, info)


def print_bound_frame(frame, info: sensel.SenselSensorInfo):
    global left_bound, up_bound, right_bound, down_bound
    rows = info.num_rows
    cols = info.num_cols
    for i in range(0, info.num_rows):
        for j in range(0, info.num_cols):
            if (frame.force_array[i * cols + j] > 0.1):
                left_bound = min(left_bound, j)
                up_bound = min(up_bound, i)
                right_bound = max(right_bound, j)
                down_bound = max(down_bound, i)


def print_each_frame(frame, info: sensel.SenselSensorInfo):
    global frame_each
    rows = info.num_rows
    cols = info.num_cols
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            frame_each[i - UP_BOUND][j -
                                     LEFT_BOUND] = frame.force_array[i * cols +
                                                                     j]


def print_max_frame(frame, info: sensel.SenselSensorInfo):
    global sum_frame
    rows = info.num_rows
    cols = info.num_cols
    max_point = (0, 0)
    max_pressure = -1.0
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            # sum_frame[i][j] += frame.force_array[i * cols + j]
            if (frame.force_array[i * cols + j] > max_pressure):
                max_pressure = frame.force_array[i * cols + j]
                max_point = (i, j)
            # if (frame.force_array[i * cols + j] > 0.0):
            #     left_bound = min(left_bound, j)
            #     up_bound = min(up_bound, i)
    sum_frame[max_point[0]][max_point[1]] += 1.0


def print_sum_frame(frame, info: sensel.SenselSensorInfo):
    global sum_frame
    rows = info.num_rows
    cols = info.num_cols
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            sum_frame[i][j] += frame.force_array[i * cols + j]


def save_frame(frame, info: sensel.SenselSensorInfo):
    global frame_series
    rows = info.num_rows
    cols = info.num_cols
    fs = np.zeros((DOWN_BOUND + 1 - UP_BOUND, RIGHT_BOUND + 1 - LEFT_BOUND))
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            fs[i - UP_BOUND][j - LEFT_BOUND] += frame.force_array[i * cols + j]
    frame_series.append(fs)


def plot_frame():
    global sum_frame
    plt.imshow(sum_frame, cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
    sum_frame = np.zeros((HEIGHT, WIDTH))


def plot_avg_frame():
    global frame_series
    from Plot import plotOneLettersCorner8
    plotOneLettersCorner8(frame_series)


def plot_each_frame():
    plt.clf()
    plt.imshow(frame_each, cmap=plt.cm.hot, vmin=0, vmax=100)
    plt.colorbar()
    plt.pause(0.01)


def getAveragePic(path):
    pic = np.zeros((70, 100))
    for frame in path:
        sum_force = 0
        x_average = 0
        y_average = 0
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                sum_force += frame[y_coordinate][x_coordinate]
        if (sum_force > 100):
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate
                    y_average += rate * y_coordinate
            try:
                pic[int(x_average * 10) - 100][int((35 - y_average) * 10) -
                                               150] = 1
            except:
                pass
    return pic


def close_sensel(frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)


def feedback_record():
    global current_record_num, plotting, frame_series
    while not interrupted:
        if plotting:
            np.save(
                save_dir + '/' + candidates[candidate_index] + "_" +
                str(current_record_num) + ".npy", frame_series)
            frame_series = []
            current_record_num += 1
            plotting = False
        if (candidate_index >= len(candidates)):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a",
                        "--average",
                        action="store_true",
                        help="print sum of average points of each frame")
    parser.add_argument("-b",
                        "--bound",
                        action="store_true",
                        help="print record bounds")
    parser.add_argument("-f",
                        "--feedback",
                        action="store_true",
                        help="record frames with visual feedback")
    parser.add_argument("-i",
                        "--interactive",
                        action="store_true",
                        help="print interactive realtime pressure")
    parser.add_argument("-m",
                        "--max",
                        action="store_true",
                        help="print sum of max points of each frame")
    parser.add_argument("-n",
                        "--name",
                        help="specify the name of the saving directory")
    parser.add_argument("-p", "--predict", help="print predicted input")
    parser.add_argument("-r",
                        "--record",
                        action="store_true",
                        help="record frames")
    parser.add_argument("-s",
                        "--sum",
                        action="store_true",
                        help="print sum of each frames")

    args = parser.parse_args()
    if args.average:
        frame_operator = save_frame
    elif args.bound:
        frame_operator = print_bound_frame
    elif args.feedback:
        frame_operator = save_frame
    elif args.interactive:
        plt.ion()
        recording = True
        frame_operator = print_each_frame
    elif args.max:
        frame_operator = print_max_frame
    elif args.predict:
        frame_operator = save_frame
    elif args.record:
        frame_operator = save_frame
    elif args.sum:
        frame_operator = print_sum_frame
    else:
        print("Please enter the mode. Or check the list by -h")
        exit(0)

    save_dir = "new_data/ch_data"
    if not args.name is None:
        save_dir = save_dir + "_" + args.name
    else:
        save_dir = save_dir + "_" + str(int(time.time()))
    if args.feedback or args.record:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    if args.predict:
        from keras.preprocessing.sequence import pad_sequences
        from keras.models import load_model
        if args.predict == 'cnn':
            model = load_model('deal-with-data/model/cnn_model.h5')
        elif args.predict == 'rnn':
            model = load_model('deal-with-data/model/rnn_model.h5')
        else:
            print("Please choose your model [cnn|rnn]")
            exit(0)

    handle = open_sensel()
    if handle:
        error, info = sensel.getSensorInfo(handle)
        frame = init_frame()

        u = threading.Thread(target=scan_frames, args=(frame, info))
        u.setDaemon(True)
        u.start()

        # t = threading.Thread(target=wait_for_enter)
        # t.setDaemon(True)
        # t.start()

        if args.interactive:
            while not interrupted:
                plot_each_frame()
        elif args.feedback:
            f = threading.Thread(target=feedback_record)
            f.setDaemon(True)
            f.start()

            app = pg.mkQApp()
            win = FeedbackRecordWindow()
            win.resize(560, 700)
            win.show()
            app.exec_()
        else:
            while True:
                if plotting:
                    code = input(
                        'Press Enter to stop... Or \'p\' to return to previous'
                    )
                    recording = False
                    plotting = False
                    if args.bound:
                        print(left_bound, right_bound, up_bound, down_bound)
                    elif args.max:
                        plot_frame()
                    elif args.predict:
                        try:
                            if args.predict == 'cnn':
                                x = getAveragePic(np.array(frame_series))
                                x = x.reshape(1, 70, 100, 1)
                                ans_id = np.argmax(model.predict(x),
                                                   axis=-1)[0]
                                # print(model.predict(x))
                                print(candidates[ans_id])
                            elif args.predict == 'rnn':
                                frame_series = np.array([
                                    frame_series
                                ]).reshape(-1, len(frame_series), 980)
                                # frame_series[0] = frame_series[0].reshape(-1, 980)
                                frame_series = pad_sequences(frame_series,
                                                             maxlen=MAX_LEN,
                                                             dtype="float32")
                                ans_id = np.argmax(model.predict(frame_series),
                                                   axis=-1)[0]
                                print(model.predict(frame_series))
                                print(candidates[ans_id])
                        except Exception as e:
                            # print("Deprecated data. Please try again.")
                            print(traceback.format_exc())
                    elif args.record:
                        np.save(
                            save_dir + "/" + candidates[candidate_index] +
                            "_" + str(current_record_num) + ".npy",
                            frame_series)
                    elif args.sum:
                        plot_frame()
                    elif args.average:
                        plot_avg_frame()
                    if code == 'q':
                        interrupted = True
                        break
                    if code != 'p':
                        current_record_num += 1
                    if current_record_num >= 5:
                        current_record_num = 0
                        candidate_index += 1
                    if candidate_index >= 26:
                        interrupted = True
                        break

                    frame_series = []
                else:
                    code = input('Press Enter to start record of ' +
                                 candidates[candidate_index])
                    if (code == 'q'):
                        interrupted = True
                        if args.feedback:
                            QtCore.QCoreApplication.instance().quit()
                        break
                    recording = True
                    plotting = True
                if (candidate_index >= len(candidates)):
                    break
        close_sensel(frame)