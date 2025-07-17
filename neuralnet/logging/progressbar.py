import numpy as np


class ProgressBar:
    def __init__(self, length : int, total_steps : int):
        self.length = length
        self.total_steps = total_steps
        self.step = 0

        self.isFull = False

    def __str__(self):
        percentage = self.step / self.total_steps
        progress = int(np.ceil(percentage * self.length))
        return "[" + '=' * (progress - 1) + ">" + "." * (self.length - progress) + "]"
    def update(self, step : int):
        step_prev = self.step
        self.step = step

        percentage_curr = self.step / self.total_steps
        progress_curr = int(percentage_curr * self.length)

        percentage_prev = step_prev / self.total_steps
        progress_prev = int(percentage_prev * self.length)

        self.isFull = self.step >= self.total_steps - 1

        return progress_curr >= progress_prev

    def clear(self):
        self.step = 0