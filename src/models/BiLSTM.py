from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sentence_classifier.preprocessing.tokenisation.tokeniser import parse_tokens
from sentence_classifier.preprocessing.reader import load


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


if __name__ == "__main__":
    """
    Train BiLSTM as an auto encoder.
    """
    questions, classifications = load("../../data/train.txt")
    tokenised_questions = list(map(lambda x: parse_tokens(x), questions))

    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    # For each words-list (sentence) and tags-list in each tuple of training_data
    word_to_ix = {}
    for sent in tokenised_questions:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

    EMBEDDING_DIM = HIDDEN_DIM = 300
    model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(word_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    average_loss = deque(maxlen=500)

    for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
        for i, sentence in enumerate(tokenised_questions):
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, sentence_in)
            average_loss.append(loss.detach().numpy())

            loss.backward()
            optimizer.step()

            print(f"\rEpoch: {epoch} {i}/{len(tokenised_questions)} Loss: {np.mean(average_loss)}", end="")

    torch.save(model.state_dict(), "../../data/BiLSTM_auto_encoder.pth")