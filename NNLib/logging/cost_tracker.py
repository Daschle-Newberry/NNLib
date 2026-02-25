import numpy as np
class CostTracker:
    def __init__(self, epsilon : float, alpha: int):
        self.cost_history = [];
        self.epsilon = epsilon
        self.last_avg = 0
        self.alpha = alpha;


    def __str__(self):
        return f"Loss : {round(self.last_avg,4)}"

    def update(self, **kwargs):
        cost = kwargs.get("cost")

        if(not len(self.cost_history) < self.alpha): self.cost_history.pop();

        self.cost_history.append(cost);
        avg = sum(self.cost_history)/len(self.cost_history)

        if self.last_avg + self.epsilon <= avg or avg <=  self.last_avg - self.epsilon:
            self.last_avg = avg
            return True

        return False


