from dataclasses import dataclass

@dataclass
#class Result:
#    batch_number: int
#    num_hits: int

class Result:
    def __init__(self, batch_number, **kwargs):
        self.batch_number = batch_number
        self.metrics = kwargs
