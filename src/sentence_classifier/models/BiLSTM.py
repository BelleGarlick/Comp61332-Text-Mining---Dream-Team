from collections import deque
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sentence_classifier.preprocessing.tokenisation.tokeniser import parse_tokens
from sentence_classifier.preprocessing.reader import load


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,
                 bidirectional: Optional[bool] = True,
                 combine_hidden: Optional[str] = "sum"):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.hidden_state_combiner = self.hidden_state_adder_fn() if combine_hidden == "sum" else self.hidden_state_concat_fn
        self.output_dim = hidden_dim*2 if bidirectional and combine_hidden != "sum" else hidden_dim

    @staticmethod
    def hidden_state_adder_fn() -> Callable[[torch.FloatTensor], torch.FloatTensor]:
        adder_fn = lambda hidden_states_tensor: hidden_states_tensor[0,:,:] + hidden_states_tensor[1,:,:]
        return adder_fn

    @staticmethod
    def hidden_state_concat_fn() -> Callable[[torch.FloatTensor], torch.FloatTensor]:
        concat_fn = lambda hidden_states_tensor: torch.cat([hidden_states_tensor[0,:,:], hidden_states_tensor[1,:,:]])
        return concat_fn

    def get_final_hidden_state(self, hidden_states: torch.LongTensor) -> torch.LongTensor:
        """
        Extracts the final hidden state, which is what we want to use as the sentence representation
        :param hidden_states:
        :return: the final hidden_state in the complicated hidden_state tensor returned after a pass through nn.LSTM()
        """
        num_layers = self.lstm.num_layers
        num_directions = 2 if self.lstm.bidirectional else 1
        batch_size = 1 # TODO: fix once we support batched inputs
        hidden_size = self.hidden_dim

        final_hidden_state = hidden_states.view(num_layers, num_directions, batch_size, hidden_size)
        return final_hidden_state

    def forward(self, sentence_word_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        :param sentence_word_embeddings: A 3D tensor with dims (batch_size, embedding_size, sequence_length)
        representing a batch of sentences that have each been transformed into a (padded) sequence of word
        embeddings
        :return: A 2D tensor with dims (batch_size, sentence_embedding_size) representing the batch of word-embedding
        sentences each transformed into a single vector (sentence representation)
        """
        # need to reshape these single inputs to be "batches" with size 1
        # TODO: undo the below reshaping once we have everything coming in already batched
        # reshaped = sentence_word_embeddings.view([1] + list(sentence_word_embeddings.size()))
        reshaped = sentence_word_embeddings
        output, (hn, cn) = self.lstm(reshaped)
        final_hidden_state = self.hidden_state_combiner(hn)
        return final_hidden_state


if __name__ == "__main__":
    """
    Train BiLSTM as an auto encoder.
    """
    questions, classifications = load("../../../data/train.txt")
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