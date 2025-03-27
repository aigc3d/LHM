import torch.nn as nn


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


class LinerParameterTuner:
    def __init__(self, start, start_value, end_value, end):
        self.start = start
        self.start_value = start_value
        self.end_value = end_value
        self.end = end
        self.total_steps = self.end - self.start

    def get_value(self, step):
        if step < self.start:
            return self.start_value
        elif step > self.end:
            return self.end_value

        current_step = step - self.start

        ratio = current_step / self.total_steps

        current_value = self.start_value + ratio * (self.end_value - self.start_value)
        return current_value


class StaticParameterTuner:
    def __init__(self, v):
        self.v = v

    def get_value(self, step):
        return self.v
