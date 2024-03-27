from dataclasses import dataclass

@dataclass
class Result:
    batch_number: int
    num_hits: int
    positive_class_count_in_unseen: int
