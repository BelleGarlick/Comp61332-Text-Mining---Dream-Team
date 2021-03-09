from unittest import TestCase
from torch import nn
from torch.tensor import Tensor
from torch.nn import Embedding
from torch import FloatTensor
from torch.nn.utils.rnn import pad_sequence

import torch
import numpy as np

from typing import List, Iterable, Dict

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
        # TODO: this needs to take a 2d IntTensor/LongTensor as input with dimensions (batch_size, padded_sentence_length)
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
        self.label_dict: Dict[str, int] = {}

        for idx, label in enumerate(set(labels)):
            self.label_dict[label] = idx

    def one_hot_vec_for(self, label: str) -> torch.LongTensor:
        vec = np.zeros(len(self.label_dict))
        pos = self.label_dict[label]
        vec[pos] = 1
        return torch.LongTensor(vec)

    def idx_for_label(self, label: str) -> int:
        return self.label_dict[label]

    def label_for_idx(self, idx: int) -> str:
        return list(self.label_dict.keys())[idx]


class EndToEndTest(TestCase):

    def generate_training_batch(self, questions: List[str]) -> Tensor:
        """
        Given a list of questions, returns a Tensor of the questions stacked and padded
        :param questions:
        :return:
        """
        # TODO: need to convert each question for List[str] to List[int] i.e list of corresponding indexes in the vocabulary
        return pad_sequence(questions)

    def test_end_to_end(self):
        torch.manual_seed(42)

        test_model = TestModel()

        lr = 1e-1
        loss_fn = nn.NLLLoss(reduction="mean")
        optimizer = torch.optim.SGD(test_model.parameters(), lr=lr)

        training_data_file_path = "../data/train.txt"
        questions, labels = load(training_data_file_path)
        one_hot_labels = OneHotLabels(labels)

        epochs = 10
        for epoch in range(epochs):
            for count in range(len(questions)):
                test_model.train()

                question = questions[count]
                label = labels[count]

                yhat = test_model(parse_tokens(question))

                loss = loss_fn(yhat.reshape(1, 50), torch.LongTensor([one_hot_labels.idx_for_label(label)]))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # do a test on the trained model (might not always work but hopefully should always work with the random seed)
        test_dataset_file_path = "../data/test.txt"
        test_questions, test_labels = load(test_dataset_file_path)

        correct_predictions = 0
        for count in range(len(test_questions)):
            test_question = questions[count]
            test_label = labels[count]

            predicted_log_probabilities = test_model(parse_tokens(test_question))
            predicted_label = one_hot_labels.label_for_idx(torch.argmax(predicted_log_probabilities))

            correct_predictions = correct_predictions + 1 if predicted_label == test_label else correct_predictions

        self.assertTrue(correct_predictions/len(test_questions) >= 0.5)  # i.e assert at least 50% accuracy
