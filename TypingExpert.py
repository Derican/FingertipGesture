import json, os
import random
import socket
import string
import threading
import sys
import time
import traceback

from utils import predictAllInOne, saveFramesAsPath

sys.path.append('sensel-lib-wrappers/sensel-lib-python')
import sensel
import numpy as np
import matplotlib.pyplot as plt
import argparse
from queue import Queue

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
    global recording
    while not interrupted:
        error = sensel.readSensor(handle)
        error, num_frames = sensel.getNumAvailableFrames(handle)
        for i in range(num_frames):
            error = sensel.getFrame(handle, frame)
            if recording:
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


PANGRAM_SENTENCES = [
    'waltz bad nymph for quick jigs vex', 'quick zephyrs blow vexing daft jim',
    'sphinx of black quartz judge my vow'
]
MACKENZIE_SENTENCES = [
    'breathing is difficult', 'my favorite subject is psychology',
    'circumstances are unacceptable', 'world population is growing',
    'earth quakes are predictable', 'correct your diction immediately',
    'express delivery is very fast', 'the minimum amount of time',
    'luckily my wallet was found', 'apartments are too expensive'
]
PRESSURE_THRESHOLD = 10
FRAME_WINDOW = 80
BLOCK_NUM = 12

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--name',
        default='test',
        help='specify the name of participant, default as "test"')

    args = parser.parse_args()

    if args.name == 'test':
        save_dir = "study3/test"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            for i in range(BLOCK_NUM):
                os.mkdir(save_dir + '/%d' % i)
    else:
        save_dir = "study3/%s_expert" % args.name
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            for i in range(BLOCK_NUM):
                os.mkdir(save_dir + '/%d' % i)

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

        block_id = 0
        results = {}
        sentences = random.sample(MACKENZIE_SENTENCES, 1)

        while block_id < BLOCK_NUM:
            current_letter_id = 0
            current_sentence_id = 0
            words_count = 0
            for sentence in sentences:
                words_count += len(sentence.split(' '))

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
            starts_and_ends = []
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
                    while not frame_series.empty():
                        frame_series.get()
                    recording = True
                    inner_start = time.time()
                    while np.sum(frame_series.get()) <= PRESSURE_THRESHOLD:
                        pass
                    frames = []
                    while np.sum(
                            fs := frame_series.get()) > PRESSURE_THRESHOLD:
                        frames.append(fs)
                    recording = False
                    inner_end = time.time()
                    if len(frames) < FRAME_WINDOW:
                        continue
                    c = predictAllInOne(frames)
                    if c is None:
                        continue
                    saveFramesAsPath(
                        frames, save_dir + '/%d/%f_%f_%c.npy' %
                        (block_id, inner_start, inner_end, current_letter))
                    send(json.dumps({"CA": c}))
                    elapsed += inner_end - inner_start
                    if c == current_letter:
                        top_1 += 1
                    current_letter_id += 1
                    total += 1
                end = time.time()
                starts_and_ends.append({'start': start, 'end': end})

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
            print("wpm: %f" % ((total - 1) * 12 / elapsed))
            results[block_id] = {
                'block_id': block_id,
                'sentences': sentences,
                'words_count': words_count,
                'total_letters': total,
                'accurate_letters': top_1,
                'accuracy': (top_1 / total),
                'starts_and_ends': starts_and_ends,
                'wpm': ((total - 1) * 12 / elapsed)
            }
            time.sleep(1)
            block_id += 1
        with open('study3/%s_expert/meta.json' % args.name, 'w') as f:
            json.dump(results, f)
        send(
            json.dumps({
                'BR': "NaN",
                'IR': "You have completed all the blocks. Thank you!",
                'TR': "",
                "CR": ""
            }))
        close_sensel(frame)