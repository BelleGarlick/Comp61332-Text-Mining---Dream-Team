from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class BagOfWords:
    sentence_representation: np.ndarray

    @staticmethod
    def from_word_embeddings(word_embeddings: List[np.ndarray]) -> 'BagOfWords':
        return BagOfWords(sum(word_embeddings) / len(word_embeddings))
