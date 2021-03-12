from unittest import TestCase
from torch import nn


import torch
import numpy as np

from typing import List, Iterable, Dict, Any, Generator, Tuple

from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation import parse_tokens
from sentence_classifier.models.model import Model


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

    def batch_training_data(self,
                            training_data: List[Any],
                            training_labels: List[Any],
                            batch_size: int) -> Generator[Tuple[List[Any], List[Any]], None, None]:
        """
        Given a dataset of training examples and labels, generates batches of desired size as a tuples of lists i.e
        yields/generates (training_data_batch, training_label_batch) until the dataset is exhausted
        :param training_data:
        :param training_labels:
        :param batch_size:
        :return: a generator that yields/streams example-label batches in a tuple
        """
        if batch_size > len(training_data):
            raise
        else:
            curr_index = 0
            training_data_size = len(training_data)

            while curr_index < training_data_size:
                next_batch = (training_data[curr_index:curr_index+batch_size],
                              training_labels[curr_index:curr_index+batch_size])
                curr_index += batch_size
                yield next_batch

    def test_end_to_end(self):
        torch.manual_seed(42)

        test_model = (Model.Builder()
                      .with_glove_word_embeddings("../data/glove.small.txt")
                      .with_bow_sentence_embedder()
                      .with_classifier(300)
                      .build())

        lr = 1.6e-1
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

        accuracy = correct_predictions/len(test_questions)
        print(f'End-to-end test accuracy: {accuracy * 100}%')
        self.assertTrue(accuracy >= 0.5)  # i.e assert at least 50% accuracy

    def test_end_to_end_bilstm(self):
        torch.manual_seed(42)

        test_model = (Model.Builder()
                      .with_glove_word_embeddings("../data/glove.small.txt")
                      .with_bilstm_sentence_classifier(300, 300)
                      .with_classifier(300)
                      .build())

        lr = 0.1
        loss_fn = nn.NLLLoss(reduction="mean")
        optimizer = torch.optim.SGD(test_model.parameters(), lr=lr)

        training_data_file_path = "../data/train.txt"
        questions, labels = load(training_data_file_path)
        one_hot_labels = OneHotLabels(labels)

        epochs = 10
        for epoch in range(epochs):
            for count in range(len(questions)):
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

        accuracy = correct_predictions / len(test_questions)
        print(f'End-to-end test accuracy: {accuracy * 100}%')
        self.assertTrue(accuracy >= 0.5)  # i.e assert at least 50% accuracy

    def test_end_to_end_test_bow_random_embeddings(self):
        torch.manual_seed(42)

        training_data_file_path = "../data/train.txt"

        test_model = (Model.Builder()
                      .with_random_word_embeddings(training_data_file_path, 300)
                      .with_bow_sentence_embedder()
                      .with_classifier(300)
                      .build())

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

        accuracy = correct_predictions / len(test_questions)
        print(f'End-to-end test accuracy: {accuracy * 100}%')
        self.assertTrue(accuracy >= 0.35)  # i.e assert at least 35% accuracy (due to frozen and random word-embeddings)
