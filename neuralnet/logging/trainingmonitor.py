from neuralnet.logging.counter import Counter
from neuralnet.logging.progressbar import ProgressBar


class TrainingMonitor:
    def __init__(self, epochs : int, batches : int):
        self.counter = Counter("Epoch",epochs)
        self.progress_bar = ProgressBar(10, batches)
        self.loss_total = 0
        self.updates = 0
        self.loss_avg_prev = 0


    def update(self, epoch : int, batch : int, loss : float):
            self.loss_total += loss
            self.updates += 1

            loss_avg = self.loss_total / self.updates
            if self.counter.update(epoch):
                print(self.counter)

            if self.progress_bar.update(batch) or self.loss_avg_prev + 1E-2 <= loss_avg or self.loss_avg_prev - 1E-2 >= loss_avg:
                print(f"\r{self.progress_bar} Loss : {round(loss_avg,4)}", end = '')
            if self.progress_bar.isFull:
                print()

            self.loss_avg_prev = loss_avg