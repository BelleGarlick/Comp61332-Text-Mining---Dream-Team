import numpy as np
from typing import List, Optional, Dict


"""
This module contains the one hot encoder class used to encode text.

This module's class allows the list of sentences to be encoded as a one hot vector for each token in the sentence.


Usage:
    encoder = OneHotEncoder()
    X = encoder.encode(train_x, update_corpus=True)
    test_X = encoder.encode(test_x)

"""


# Place holder for any new word not in corpus.
UNKNOWN_TOKEN = "#UNK#"


class OneHotEncoder:
    def __init__(self, mapping: Optional[Dict] = None):
        """
        Initialise the one hot encoder class.

        This object, when constructed will initialise a blank corpus with only the token #unk# included. Any vector
        with an unknown word will use this as the index in the one hot array.
        """
        self.corpus = {
            UNKNOWN_TOKEN: 0
        }
        self.mapping = mapping if mapping is not None else {}  # TODO: verify how self.mapping is used

    def encode(self, data: List[List[str]], update_corpus=False):
        """
        One hot encode the data.

        This function, given a list of sentences, will convert said sentences to a one hot encoding. If the
        update_corpus bool is enabled then any new words will be added to the corpus.

        Args:
            data: List of all sentences made up of a list of tokens. This data will be one hot encoded.
            update_corpus: Should new words be added to the corpus.

        Returns:
            A list of matricies which are the concatination of one hot vectors.
            [
                np.array([
                    [one hot],
                    [one_hot]
                ]),
                np.array([
                    [one hot],
                    [one_hot]
                ])
            ]
        """
        if update_corpus:
            self.__populate_dictionary(data)

        vector_length = len(self.corpus)

        one_hot_questions = []
        for question in data:
            one_hot_tokens = []
            for token in question:
                # If not in corpus, set to unknown token.
                token = token if token in self.corpus else UNKNOWN_TOKEN

                # Create and update one hot vector
                one_hot_vector = np.zeros(vector_length)
                one_hot_vector[self.corpus[token]] = 1
                one_hot_tokens.append(one_hot_vector)

            one_hot_questions.append(np.concatenate(
                np.expand_dims(one_hot_tokens, axis=0)
            ))

        return one_hot_questions

    def __populate_dictionary(self, data: List[List[str]]):
        """
        Populate the dictionary with new terms.

        For each token in each sentence in the data, if the token is new it shall be added to the mapping vocabulary.
        This mapping converts a given token and maps it to a specific integer which represents which index in the one
        hot encoding should be a '1'.

        Args:
            data: The list of sentences to add new words from.
        """
        for question in data:
            for token in question:
                if token not in self.corpus:
                    # We can get the new index simply by choosing the current length of the mapping dictionary
                    #     e.g. If we have no items in the dict, then the first item will be at index 0.
                    self.corpus[token] = len(self.corpus)
