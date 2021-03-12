from typing import Set
from collections.abc import Iterable

from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation.tokeniser import parse_tokens


class VocabUtils:

    @staticmethod
    def vocab_from_training_data(training_data_file_path: str) -> Set[str]:
        questions, _ = load(training_data_file_path)
        vocab = VocabUtils.vocab_from_text_corpus([parse_tokens(question) for question in questions])
        return vocab

    @staticmethod
    def vocab_from_text_corpus(test_corpus: Iterable) -> Set[str]:
        """
        Given a corpus (represented as a List of str or a List-of-List of str, etc.), returns a set of unique
        words that occur
        :param test_corpus:
        :return: The set of unique words in the coru
        """
        return set(VocabUtils.flatten(test_corpus))

    @staticmethod
    def flatten(l):
        """
        Flattens an arbitrarily nested sequence of sequences. Stolen from https://stackoverflow.com/a/2158532
        :param l:
        :return:
        """
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from VocabUtils.flatten(el)
            else:
                yield el