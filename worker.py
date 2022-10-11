from abc import abstractmethod
from constants import *


class Worker:

    def __init__(self, target_set) -> None:
        self.target_set = target_set

    def run(self):
        with Pool(6) as pool:
            result = pool.map(self.operator, self.target_set)
            return result

    def runSingle(self):
        for target in self.target_set:
            self.operator(target)
        return None

    @abstractmethod
    def operator(self, person):
        pass