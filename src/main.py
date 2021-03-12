import argparse
import sys


from sentence_classifier.utils.config import Config

from sentence_classifier.preprocessing.dataloading import DatasetQuestions
from sentence_classifier.preprocessing.reader import load
from sentence_classifier.models.model import Model
from sentence_classifier.utils.one_hot_labels import OneHotLabels
from sentence_classifier.preprocessing.tokenisation import parse_tokens

from torch.utils.data import DataLoader
import torch

from typing import Callable


# Rules used during tokenisation.
tokenisation_rules = {
    "TOKENISE_QUOTES": True,
    "TOKENISE_NUMBERS": True,
    "TOKENISE_MONTH": True,
    "TOKENISE_PERCENTAGES": True,
    "REMOVE_QUESTION_MARKS": True,
    "TOKENISE_STOPWORDS": False,
    "TOKENISE_URLS": True,
    "TOKENISE_MONEY": True,
    "TOKENISE_YEAR": True,
    "TOKENISE_COMMA_SEPERATED_NUMBERS": True
}


def train_model(model: Model, training_data_file_path: str, loss_fn: Callable,
                num_epochs: int, optimizer: torch.optim.Optimizer):

    torch.manual_seed(42)
    model.train()

    questions, labels = load(training_data_file_path)
    one_hot_labels = OneHotLabels.from_labels_json_file("../data/labels.json")

    for epoch in range(num_epochs):
        for count in range(len(questions)):
            question = questions[count]
            label = labels[count]

            yhat = model(parse_tokens(question))

            loss = loss_fn(yhat.reshape(1, 50), torch.LongTensor([one_hot_labels.idx_for_label(label)]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.train(False)


def test_model(model: Model, test_dataset_file_path: str) -> float:
    # TODO: report model RoC metrics instead of just accuracy
    """
    Given a trained model, runs it against a test dataset and reports the accuracy
    :param model:
    :return:
    """
    one_hot_labels = OneHotLabels.from_labels_json_file("../data/labels.json")
    test_questions, test_labels = load(test_dataset_file_path)

    correct_predictions = 0
    for count in range(len(test_questions)):
        test_question = test_questions[count]
        test_label = test_labels[count]

        predicted_log_probabilities = model(parse_tokens(test_question))
        predicted_label = one_hot_labels.label_for_idx(torch.argmax(predicted_log_probabilities))

        correct_predictions = correct_predictions + 1 if predicted_label == test_label else correct_predictions

    accuracy = correct_predictions / len(test_questions)
    print(f'End-to-end test accuracy: {accuracy * 100}%')
    return accuracy


def save_model(model: Model, save_model_file_path: str) -> str:
    torch.save(model, save_model_file_path)
    return save_model_file_path


def load_model(save_model_file_path: str) -> Model:
    return torch.load(save_model_file_path)


class ArgException(Exception):
    pass


if __name__ == "__main__":
    #Command line arguments parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run the code to train a model')
    parser.add_argument('--test', action='store_true', help='Run the code to test a model')
    parser.add_argument('--config', nargs='?', default='../data/config.ini')
    args = parser.parse_args(sys.argv[1:])

    config_file = args.config
    config = Config.from_config_file(config_file)

    if args.train:
        model = Config.build_model_from_config(config_file)

        train_model(model, config.path_train, torch.nn.NLLLoss(reduction="mean"),
                    config.epochs, torch.optim.Adam(model.parameters(), lr=config.lr))

        save_model(model, "../data/saved_models/model.bin")  # TODO: parameterise model save location
    elif args.test:
        model = load_model("../data/saved_models/model.bin")
        test_model(model, config.path_test)
    else:
        raise ArgException("Argument --train or --test must be passed")

    model = Config.build_model_from_config(config_file)

    train_model(model, config.path_train, torch.nn.NLLLoss(reduction="mean"),
                config.epochs, torch.optim.Adam(model.parameters(), lr=config.lr))

    # -------------------------------------------------------------------- #

    # TODO: Handle the case when the argument is --test instead of --train
    # Load the dataset
    train_dataset = DatasetQuestions(config.path_train, tokenisation_rules=tokenisation_rules, dict_path="../data/glove.small.txt")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=train_dataset.collate_fn)

    # TODO: train the model etc.
    for q, l in train_loader:
        print(q, l)
