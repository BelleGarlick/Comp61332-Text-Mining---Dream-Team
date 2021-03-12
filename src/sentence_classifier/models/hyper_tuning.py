import numpy as np
import torch
from torch import nn

from sentence_classifier.analysis import roc
from sentence_classifier.models.model import Model
from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation import parse_tokens
from test.test_end_to_end import OneHotLabels


REPEATS = 3
CLASSES = 50

TRAIN_FILE_PATH = "../data/train.txt"
VAL_FILE_PATH = "../data/dev.txt"

X, Y = load(TRAIN_FILE_PATH)
val_X, val_Y = load(VAL_FILE_PATH)

one_hot_labels = OneHotLabels(Y)


def train_model(model, loss_fn, optimizer, repeat, epochs=10):
    for epoch in range(epochs):
        avg_loss = []
        for count in range(len(X)):
            model.train()

            question = X[count]
            label = Y[count]

            yhat = model(parse_tokens(question))

            loss = loss_fn(yhat.reshape(1, CLASSES), torch.LongTensor([one_hot_labels.idx_for_label(label)]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss.append(loss.detach().numpy())

            print("\rExperiment: {}, Epochs: {}, Step: {}/{}, Loss: {}".format(
                repeat + 1, epoch + 1, count + 1, len(X), np.mean(avg_loss)
            ), end="")

    return roc.analyse(val_Y, [
        one_hot_labels.label_for_idx(torch.argmax(model(parse_tokens(test_question))))
        for test_question in val_X
    ])["f1"]


def build_model(model_type="bow"):
    if model_type == "bow":
        return (Model.Builder()
                .with_glove_word_embeddings("../data/glove.small.txt")
                .with_bow_sentence_embedder()
                .with_classifier(300)
                .build())

    elif model_type == "bilstm":
        return (Model.Builder()
                .with_glove_word_embeddings("../data/glove.small.txt")
                .with_bilstm_sentence_embedder(300, 300)
                .with_classifier(300)
                .build())

    raise Exception("Please specify a model from {'bow', 'bilstm'}")


if __name__ == "__main__":
    for lr in [0.02, 0.04, 0.06, 0.08, 0.1]:
        results = []
        for repeat in range(REPEATS):
            model = build_model("bilstm")

            loss_fn = nn.NLLLoss(reduction="mean")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            results.append(train_model(model, loss_fn, optimizer, repeat))

        print("\rLr: {}: {}".format(
            lr, np.mean(results)
        ))
