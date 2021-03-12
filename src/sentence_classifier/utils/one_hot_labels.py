from typing import Iterable, Dict

import torch
import numpy as np
import json


class OneHotLabels:

    def __init__(self, label_dict: Dict[str, int]):
        self.label_dict = label_dict

    def one_hot_vec_for(self, label: str) -> torch.LongTensor:
        vec = np.zeros(len(self.label_dict))
        pos = self.label_dict[label]
        vec[pos] = 1
        return torch.LongTensor(vec)

    def idx_for_label(self, label: str) -> int:
        return self.label_dict[label]

    def label_for_idx(self, idx: int) -> str:
        return list(self.label_dict.keys())[idx]

    @staticmethod
    def from_labels_json_file(labels_json_file_path: str) -> 'OneHotLabels':
        with open(labels_json_file_path, "r") as labels_json_file:
            label_dict = json.load(labels_json_file)
            return OneHotLabels(label_dict)

    @staticmethod
    def from_labels_sequence(labels: Iterable[str]) -> 'OneHotLabels':
        label_dict: Dict[str, int] = {}

        for idx, label in enumerate(set(labels)):
            label_dict[label] = idx

        return OneHotLabels(label_dict)
