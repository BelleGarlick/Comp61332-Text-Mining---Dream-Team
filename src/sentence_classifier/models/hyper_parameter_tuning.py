import numpy as np
from torch import nn
import torch

from sentence_classifier.analysis import roc
from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation import parse_tokens
from test.test_end_to_end import OneHotLabels, TestModelBagOfWords

CLASSES = 50


def run_test(model, optimizer, loss_fn, epochs=10):
    torch.manual_seed(42)
    np.random.seed(42)

    questions, labels = load("../data/train.txt")
    val_questions, val_labels = load("../data/dev.txt")

    def predict_target(input_question):
        parsed_tokens = parse_tokens(input_question, rules=tokeniser_rules)
        return one_hot_labels.label_for_idx(torch.argmax(model(parsed_tokens)))

    one_hot_labels = OneHotLabels(labels)

    results = []

    for epoch in range(epochs):
        loss_history = []
        for i, question in enumerate(questions):
            yhat = model(parse_tokens(question, tokeniser_rules))

            target = torch.LongTensor([one_hot_labels.idx_for_label(labels[i])])

            loss = loss_fn(yhat.reshape(1, CLASSES), target)

            loss_history.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_str = "{:.4f}".format(np.mean(loss_history))
            print(f"\rEpoch: {epoch + 1}, Loss: {loss_str}, {i + 1}/{len(questions)}", end="")

        # do a test on the trained model (might not always work but hopefully should always work with the random seed)
        predicted_outputs = list(map(predict_target, val_questions))
        analysis = roc.analyse(val_labels, predicted_outputs)
        results.append({"accuracy": analysis["accuracy"], "f1": analysis["f1"]})

        print("\rEpoch: {}, Loss: {:.4f}, Val Accuracy: {:.2f}%".format(
            epoch + 1,
            np.mean(loss_history),
            analysis['accuracy'] * 100
        ))

    return results


results_path = "../data/val_results/bow/lr/"

tokeniser_rules = {
    "TOKENISE_QUOTES": False,
    "TOKENISE_NUMBERS": False,
    "TOKENISE_MONTH": False,
    "TOKENISE_PERCENTAGES": False,
    "REMOVE_QUESTION_MARKS": False,
    "TOKENISE_URLS": False,
    "TOKENISE_MONEY": False,
    "TOKENISE_YEAR": False,
    "TOKENISE_COMMA_SEPERATED_NUMBERS": False
}

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    model = TestModelBagOfWords()

    lr = 0.0025
    name = f"{lr}"
    print(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss(reduction="mean")

    results = run_test(model, optimizer, loss_fn)

    with open(results_path + f"{name}.csv", "w+") as file:
        file.write("epoch,accuracy,f1\n")
        for i, result in enumerate(results):
            file.write(f"{i},{result['accuracy']},{result['f1']}\n")

    # import matplotlib.pyplot as plt
    # plt.title(str(lr))
    # plt.plot([i for i in range(len(results))], [result["accuracy"] for result in results])
    # plt.plot([i for i in range(len(results))], [result["f1"] for result in results])
    # plt.show()
