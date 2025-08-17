class CostTracker:
    def __init__(self, valid_change : float):
        self.cost_history = []
        self.valid_change = valid_change
        self.total_cost = 0
        self.updates = 0
        self.last_epoch = 1
        self.last_avg = 0
    def __str__(self):
        return f"Loss : {round(self.last_avg,4)}"

    def update(self, **kwargs):
        cost = kwargs.get("cost")
        epoch = kwargs.get("epoch")

        if epoch != self.last_epoch:
            self.cost_history.append(self.total_cost / self.updates)
            self.total_cost = 0
            self.last_avg = 0
            self.updates = 0
            self.last_epoch = epoch

        self.total_cost += cost
        self.updates += 1

        avg = self.total_cost / self.updates


        if self.last_avg + self.valid_change <= avg or avg <=  self.last_avg - self.valid_change:
            self.last_avg = avg
            return True

        return False


