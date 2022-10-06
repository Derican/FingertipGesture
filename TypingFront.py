import json
import sys
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QApplication,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import socket


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.functions = {
            'BR': self.replaceBlock,
            'BA': self.appendBlock,
            'IR': self.replaceInfo,
            'IA': self.appendInfo,
            'TR': self.replaceTask,
            'TA': self.appendTask,
            'CR': self.replaceCandidate,
            'CA': self.appendCandidate,
        }

        self.initUI()

    def initUI(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_th = RecvThread()
        self.s_th.sinOut.connect(self.change)
        self.s_th.start()

        block_label = QLabel(self)
        block_label.setFont(QFont("Cascadia Code", 30, QFont.Weight.Bold))
        block_label.setText("Block: NaN")
        block_label.move(100, 100)
        self.block_label = block_label
        self.block = ""

        info_label = QLabel(self)
        info_label.setFont(QFont("Cascadia Code", 30, QFont.Weight.Bold))
        info_label.setText("Info: ")
        info_label.move(100, 200)
        self.info_label = info_label
        self.info = ""

        task_label = QLabel(self)
        task_label.setFont(QFont("Cascadia Code", 30, QFont.Weight.Bold))
        task_label.setText("Task: ")
        task_label.move(100, 300)
        self.task_label = task_label
        self.task = ""

        candidate_label = QLabel(self)
        candidate_label.setFont(QFont("Cascadia Code", 30, QFont.Weight.Bold))
        candidate_label.setText("Your: ")
        candidate_label.move(100, 400)
        self.candidate_label = candidate_label
        self.candiate = "Your: "

        self.time = QTimer(self)
        self.time.setInterval(1000)
        self.time.timeout.connect(self.refresh)

        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle('Evaluation')
        self.show()

    def refresh(self):
        pass

    def replaceBlock(self, data):
        self.block = data
        self.block_label.setText("Block: " + self.block)
        self.block_label.adjustSize()

    def appendBlock(self, data):
        self.block += data
        self.block_label.setText("Block: " + self.block)
        self.block_label.adjustSize()

    def replaceInfo(self, data):
        self.info = data
        self.info_label.setText("Info:  " + self.info)
        self.info_label.adjustSize()

    def appendInfo(self, data):
        self.info += data
        self.info_label.setText("Info:  " + self.info)
        self.info_label.adjustSize()

    def replaceTask(self, data):
        self.task = data
        self.task_label.setText("Task:  " + self.task)
        self.task_label.adjustSize()

    def appendTask(self, data):
        self.task += data
        self.task_label.setText("Task:  " + self.task)
        self.task_label.adjustSize()

    def replaceCandidate(self, data):
        self.candidate = data
        self.candidate_label.setText("Your:  " + self.candidate)
        self.candidate_label.adjustSize()

    def appendCandidate(self, data):
        self.candidate += data
        self.candidate_label.setText("Your:  " + self.candidate)
        self.candidate_label.adjustSize()

    def change(self, top):
        msg = json.loads(top)
        for key, val in msg.items():
            self.functions[key](val)


class RecvThread(QThread):
    sinOut = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super(RecvThread, self).__init__(parent)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("localhost", 34826))

    def run(self):
        while True:
            data, addr = self.s.recvfrom(2048)
            top: str = data.decode()
            print("recv: ", top)
            self.sinOut.emit(top)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())