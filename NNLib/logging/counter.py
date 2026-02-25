class Counter:
    def __init__(self, name : str, limit : int, param_name : str):
        self.name = name
        self.limit = limit
        self.step = 0
        self.param_name = param_name


    def __iadd__(self, step : int):
        self.step += step
        return self

    def __str__(self):
        return f"{self.name} {self.step + 1} / {self.limit}"

    def update(self, **kwargs):
        step = kwargs.get(self.param_name)
        if self.step == step:
            return False
        else:
            self.step = step
            return True

