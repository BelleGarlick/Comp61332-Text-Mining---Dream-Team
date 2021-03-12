import numpy as np
import torch
from torch import nn

from sentence_classifier.analysis import roc
from sentence_classifier.models.model import Model
from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation import parse_tokens
from sentence_classifier.utils.one_hot_labels import OneHotLabels


REPEATS = 1
CLASSES = 50

TRAIN_FILE_PATH = "../data/train-og.txt"
TEST_FILE_PATH = "../data/test.txt"
LABELS_JSON_FILE = "../data/labels.json"

X, Y = load(TRAIN_FILE_PATH)
test_X, test_Y = load(TEST_FILE_PATH)


def load_model(save_model_file_path: str) -> Model:
    return torch.load(save_model_file_path)


class Ensemble:
    def __init__(self):
        self.models = [load_model(f"../data/saved_models/ensemble_weights/weights-{i + 1}.pth") for i in range(5)]
        self.one_hot_labels = OneHotLabels.from_labels_json_file(LABELS_JSON_FILE)

    def predict(self, question):
        average_tensors = []
        for model in self.models:
            yhat = model(parse_tokens(question))
            average_tensors.append(torch.squeeze(yhat).detach().numpy())
        averages = np.array(average_tensors)
        averages = np.mean(averages, axis=0)
        vote = np.argmax(averages)

        return self.one_hot_labels.label_for_idx(vote)


if __name__ == "__main__":
    ensemble = Ensemble()

    print(roc.analyse(test_Y, [
        ensemble.predict(test_question)
        for test_question in test_X
    ])["f1"])


    # one_hot_labels = OneHotLabels.from_labels_json_file(LABELS_JSON_FILE)
    #
    # def train_model(model, loss_fn, optimizer, epochs=20):
    #     for epoch in range(epochs):
    #         avg_loss = []
    #         for count in range(len(X)):
    #             model.train()
    #
    #             yhat = model(parse_tokens(X[count]))
    #
    #             loss = loss_fn(yhat.reshape(1, CLASSES), torch.LongTensor([one_hot_labels.idx_for_label(Y[count])]))
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #             avg_loss.append(loss.detach().numpy())
    #
    #             print("\rEpochs: {}, Step: {}/{}, Loss: {}".format(
    #                 epoch + 1, count + 1, len(X), np.mean(avg_loss)
    #             ), end="")
    #
    #     return roc.analyse(test_Y, [
    #         one_hot_labels.label_for_idx(torch.argmax(model(parse_tokens(test_question))))
    #         for test_question in test_X
    #     ])["f1"]
    #
    #
    # def bootstrap_dataset():
    #     X, Y = load(TRAIN_FILE_PATH)
    #     replacees = np.random.choice(len(X), len(X) // 5)
    #     replacements = np.random.choice(len(X), len(X) // 5)
    #     for i in range(len(replacees)):
    #         X[replacees[i]] = X[replacements[i]]
    #         Y[replacees[i]] = Y[replacements[i]]
    #
    #     return X, Y
    #
    #
    # def save_model(model: Model, save_model_file_path: str) -> str:
    #     torch.save(model, save_model_file_path)
    #     return save_model_file_path
    #
    #
    # for i in range(5):
    #     X, Y = bootstrap_dataset()
    #
    #     model = (Model.Builder()
    #                 .with_glove_word_embeddings("../data/glove.small.txt")
    #                 .with_bow_sentence_embedder()
    #                 .with_classifier(300)
    #                 .build())
    #     loss_fn = nn.NLLLoss(reduction="mean")
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    #
    #     results = train_model(model, loss_fn, optimizer)
    #
    #     save_model(model,  f"../data/saved_models/ensemble_weights/weights-{i + 1}.pth")
    #
    #     print("\rLr: {}: {}".format(i,
    #         np.mean(results)
    #     ))
