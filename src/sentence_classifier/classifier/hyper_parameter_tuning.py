import numpy as np
from torch import nn
import torch

from sentence_classifier.analysis import roc
from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation import parse_tokens
from test.test_end_to_end import OneHotLabels, TestModelBiLSTM

CLASSES = 50


def run_test(model, optimizer, loss_fn, epochs=10):
    torch.manual_seed(42)
    np.random.seed(42)

    questions, labels = load("../data/train.txt")
    val_questions, val_labels = load("../data/dev.txt")

    def predict_target(input_question):
        return one_hot_labels.label_for_idx(torch.argmax(model(parse_tokens(input_question))))

    one_hot_labels = OneHotLabels(labels)

    results = []

    for epoch in range(epochs):
        for i, question in enumerate(questions):
            print(f"\rEpoch: {epoch + 1}, {i + 1}/{len(questions)}", end="")

            yhat = model(parse_tokens(question))

            loss = loss_fn(yhat.reshape(1, CLASSES), torch.LongTensor([one_hot_labels.idx_for_label(labels[i])]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # do a test on the trained model (might not always work but hopefully should always work with the random seed)
        predicted_outputs = list(map(predict_target, val_questions))
        analysis = roc.analyse(val_labels, predicted_outputs)
        results.append({"accuracy": analysis["accuracy"], "f1": analysis["f1"]})

        print(f"\rEpoch: {epoch + 1}, Val Accuracy: {analysis['accuracy'] * 100}%")

    return results


if __name__ == "__main__":
    model = TestModelBiLSTM()

    lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss(reduction="mean")

    results = run_test(model, optimizer, loss_fn)

    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(results))], [result["accuracy"] for result in results])
    plt.plot([i for i in range(len(results))], [result["f1"] for result in results])
    plt.show()
