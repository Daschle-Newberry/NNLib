from . import Counter
from . import CostTracker
from . import ProgressBar


class TrainingMonitor:
    def __init__(self, epochs : int, batches : int):
        self.counter = Counter("Epoch",epochs,"epoch")
        self.progress_bar = ProgressBar(25, batches,"batch")
        self.cost_tracker = CostTracker(1E-5, 3)

        self.updates = 0
        self.is_silent = False

    def update(self,is_final: bool, **kwargs):
        if self.is_silent:
            return

        if self.cost_tracker.update(**kwargs) or self.progress_bar.update(**kwargs):
            print(f"\r{self.counter} {self.progress_bar} {self.cost_tracker}", end = '')

        if self.counter.update(**kwargs) or is_final:
            self.progress_bar.clear()
            print("")

    def get_cost_history(self):
        return self.cost_tracker.cost_history

    def quiet(self):
        self.is_silent = True