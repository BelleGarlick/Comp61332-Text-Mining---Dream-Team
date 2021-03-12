from typing import Set
from collections.abc import Iterable


class VocabUtils:

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