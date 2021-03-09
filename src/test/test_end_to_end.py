from unittest import TestCase
from torch import nn
from torch.tensor import Tensor
from torch.nn import Embedding
from torch import FloatTensor
from torch.nn.utils.rnn import pad_sequence

import torch
import numpy as np

from typing import List, Iterable

from sentence_classifier.classifier.classifier_nn import ClassifierNN
from sentence_classifier.preprocessing.bagofwords import BagOfWords

from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation import parse_tokens


class TestWordEmbeddings(nn.Module):

    def __init__(self, embeddings_file_path: str):
        super(TestWordEmbeddings, self).__init__()

        self.word_idx_dict = {}

        with open(embeddings_file_path, "r") as embeddings_file:
            float_str_to_float_tensor = lambda float_str: FloatTensor([float(_str) for _str in float_str.split()])
            pretrained_embeddings = []

            for idx, line in enumerate(embeddings_file):
                word_embedding_tuple = line.split("\t")
                word = word_embedding_tuple[0]
                self.word_idx_dict[word] = idx

                pretrained_embeddings.append(float_str_to_float_tensor(line.split("\t")[1]))

            self.embedding_layer = Embedding.from_pretrained(torch.stack(pretrained_embeddings))

    def idx_for_word(self, word: str) -> int:
        try:
            return self.word_idx_dict[word]
        except KeyError as e:
            return self.word_idx_dict["#UNK#"]

    def sentence_to_idx_tensor(self, sentence: List[str]) -> Tensor:
        return torch.LongTensor([self.idx_for_word(word) for word in sentence])

    def forward(self, sentence: List[str]):
        x = self.embedding_layer(self.sentence_to_idx_tensor(sentence))
        return x


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

        self.word_embeddings = TestWordEmbeddings("../data/glove.small.txt")  # TODO: use one of the word-embedding layers here
        self.sentence_embeddings = BagOfWords()  # TODO: use BagOfWords for now
        self.classifier = ClassifierNN(300)  # TODO: use ClassifierNN

    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.sentence_embeddings(x)
        x = self.classifier(x)

        return x

class OneHotLabels:

    def __init__(self, labels: Iterable[str]):
        self.label_dict = {}
        for idx, label in enumerate(set(labels)):
            self.label_dict[label] = idx

    def one_hot_vec_for(self, label: str):
        vec = np.zeros(len(self.label_dict))
        pos = self.label_dict[label]
        vec[pos] = 1
        return torch.LongTensor(vec)


class EndToEndTest(TestCase):


    def generate_training_batch(self, questions: List[str]) -> Tensor:
        """
        Given a list of questions, returns a Tensor of the questions stacked and padded
        :param questions:
        :return:
        """
        return pad_sequence(questions)


    def generate_one_hot_label(self, label: str, set_of_labels: Iterable[str]) -> Tensor:
        pass

    def test_end_to_end(self):

        test_model = TestModel()

        torch.manual_seed(42)

        # We can also inspect its parameters using its state_dict
        print(test_model.state_dict())

        lr = 1e-1
        n_epochs = 1000

        loss_fn = nn.NLLLoss(reduction="mean")
        optimizer = torch.optim.SGD(test_model.parameters(), lr=lr)

        training_data_file_path = "../data/train.txt"

        for epoch in range(n_epochs):
            test_model.train()

            questions, labels = load(training_data_file_path)
            one_hot_labels = OneHotLabels(labels)

            question_1 = questions[0]
            labels_1 = labels[0]

            # No more manual prediction!
            # yhat = a + b * x_tensor
            yhat = test_model(parse_tokens(question_1))

            loss = loss_fn(one_hot_labels.one_hot_vec_for(labels_1), yhat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(test_model.state_dict())
