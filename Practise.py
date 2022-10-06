import json
import random
import socket
import string
import threading
import sys
import time
import traceback

sys.path.append('sensel-lib-wrappers/sensel-lib-python')
import sensel
import numpy as np
import matplotlib.pyplot as plt
import argparse
from queue import Queue
from Plot import LETTER, getConfidenceQueue, plotDirections6, plotOneLettersCorner, getConfidenceQueue8, predictLetter

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

left_bound = 184
up_bound = 104
right_bound = 0
down_bound = 0

frame_series = Queue()


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
    frame_series.put(fs)


def close_sensel(frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)


SENTENCES = [
    'waltz bad nymph for quick jigs vex', 'quick zephyrs blow vexing daft jim',
    'sphinx of black quartz judge my vow', 'my watch fell in the water',
    'elections bring out the best'
]
PRESSURE_THRESHOLD = 10
FRAME_WINDOW = 10
BLOCK_NUM = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    handle = open_sensel()
    if handle:
        error, info = sensel.getSensorInfo(handle)
        frame = init_frame()

        u = threading.Thread(target=scan_frames, args=(frame, info))
        u.setDaemon(True)
        u.start()

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('localhost', 34827))

        def send(str: str):
            s.sendto(str.encode('gbk'), ('localhost', 34826))

        sentences = SENTENCES.copy()
        random.shuffle(sentences)
        block_id = 0
        current_sentence_id = 0
        current_letter_id = 0

        while block_id < BLOCK_NUM:
            total = 0
            top_1 = 0
            elapsed = 0
            send(
                json.dumps({
                    'BR': f"{block_id} / {BLOCK_NUM}",
                    'IR':
                    "You are going to start a block. Press Enter when you're ready",
                    'TR': "",
                    "CR": ""
                }))
            input()
            while current_sentence_id < len(sentences):
                current_sentence = sentences[current_sentence_id]
                send(
                    json.dumps({
                        'IR':
                        "You are going to start a sentence. Press Enter when you're ready",
                        "TR": current_sentence,
                        "CR": ""
                    }))
                input()
                print("Start: ", time.time())
                start = time.time()
                send(json.dumps({'IR': "Writing..."}))
                while current_letter_id < len(current_sentence):
                    current_letter = current_sentence[current_letter_id]
                    if current_letter not in string.ascii_letters:
                        current_letter_id += 1
                        send(json.dumps({"CA": ' '}))
                        continue
                    while np.sum(frame_series.get()) <= PRESSURE_THRESHOLD:
                        pass
                    frames = []
                    while np.sum(
                            fs := frame_series.get()) > PRESSURE_THRESHOLD:
                        frames.append(fs)
                    if len(frames) < FRAME_WINDOW:
                        continue
                    c = predictLetter(frames)
                    send(json.dumps({"CA": c}))
                    if c == current_letter:
                        top_1 += 1
                    else:
                        plotDirections6(frames, None, LETTER.index(c), 0)
                    current_letter_id += 1
                    total += 1
                elapsed += time.time() - start
                print("End:   ", time.time())
                current_sentence_id += 1
                current_letter_id = 0
                time.sleep(1)
            send(
                json.dumps({
                    'IR': 'Now you finished a block, take a rest!',
                    'TR': "",
                    "CR": ""
                }))
            print("total: %d" % total)
            print("acc: %f" % (top_1 / total))
            print("wpm: %f" % (total / (elapsed / 60)))
            input()
            block_id += 1
            current_letter_id = 0
            current_sentence_id = 0

        close_sensel(frame)