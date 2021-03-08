from unittest import TestCase
from sentence_classifier.preprocessing.bagofwords import BagOfWords
from typing import List
import numpy as np


class ConfigParserTest(TestCase):

    @staticmethod
    def one_hot_encode(vocab: List[str], sentence: List[str]) -> List[np.ndarray]:
        def one_hot_vector(pos: int, size: int) -> np.ndarray:
            vec = np.zeros(size)
            vec[pos] = 1
            return vec

        vocab_dict = dict([(word, i) for i, word in enumerate(vocab)])
        return [one_hot_vector(vocab_dict[word], len(vocab))
                for word in sentence]

    # TODO: there's not much to test for bag-of-words since all it does is sum and averages some vectors.
    # TODO: just have this here more to demonstrate to ourselves how bag-of-words should word
    def test_bagofwords_one_hot_encoding(self):
        """
        assumes one-hot-encoded word embeddings; these assertions wouldn't hold for word embeddings like glove/word2vec
        """
        vocab = ["animal", "car", "what", "is", "your", "name", "the", "hello", "of"]
        sentence = ["what", "is", "your", "name"]
        one_hot_encodings = self.one_hot_encode(vocab, sentence)
        bag_of_words = BagOfWords.from_word_embeddings(one_hot_encodings)

        self.assertTrue(len(set(bag_of_words.sentence_representation)) == 2)  # i.e assert 0 and some other value, makes up this vector

        sentence = ["what", "is", "the", "name", "of", "the", "animal"]
        one_hot_encodings = self.one_hot_encode(vocab, sentence)
        bag_of_words = BagOfWords.from_word_embeddings(one_hot_encodings)

        self.assertTrue(len(set(bag_of_words.sentence_representation)) == 3) # i.e assert 0, some value, and some other for multiple "the", makes up this vector

    def test_bagofwords(self):
        word_embeddings = [np.array([1, 5, 8, 3.6, 4.445]),
                           np.array([4, 6, 3.3, 3.332, 1]),
                           np.array([4.4, 27.67, 2, 3.8, 112])]

        bag_of_words = BagOfWords.from_word_embeddings(word_embeddings)
        self.assertTrue(np.array_equal(bag_of_words.sentence_representation,
                                       sum(word_embeddings) / len(word_embeddings)))
