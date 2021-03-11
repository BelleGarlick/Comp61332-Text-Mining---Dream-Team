import numpy as np
from torch import nn

import torch


from typing import Iterable, Dict, List


class WordEmbeddings(nn.Module):

    def __init__(self, vocab: Iterable[str], embeddings: List[torch.FloatTensor]):
        super(WordEmbeddings, self).__init__()

        self.vocab = vocab
        self.word_idx_dict = self.construct_word_idx_dict(vocab)
        self.embedding_layer = nn.Embedding.from_pretrained(torch.stack(embeddings))

    @staticmethod
    def from_embeddings_file(embeddings_file_path: str) -> 'WordEmbeddings':
        """
        Assumes that the embedding file is in tab-separated format, with words in the first column and
        embedding vectors in the second
        :param embeddings_file_path:
        :return: a WordEmbeddings model/layer that uses the vocab and embeddings in the provided file
        """
        with open(embeddings_file_path, "r") as embeddings_file:
            float_str_to_float_tensor = lambda float_str: torch.FloatTensor([float(_str) for _str in float_str.split()])
            pretrained_embeddings = []
            vocab = []

            for idx, line in enumerate(embeddings_file):
                word_embedding_tuple = line.split("\t")
                word = word_embedding_tuple[0]

                vocab.append(word)
                pretrained_embeddings.append(float_str_to_float_tensor(line.split("\t")[1]))

        return WordEmbeddings(vocab, pretrained_embeddings)

    @staticmethod
    def from_random_embedding(vocab: Iterable[str], emb_dim: int) -> 'WordEmbeddings':
        """
        This uses the provided vocab and creates randomly-initialised embeddings for each word
        :param emb_dim:
        :param vocab:
        :return: a WordEmbeddings model/layer that uses the provided vocab with random
        """

        random_embeddings = [torch.FloatTensor(np.random.uniform(size=emb_dim)) for word in vocab]
        return WordEmbeddings(vocab, random_embeddings)

    @staticmethod
    def construct_vocab_from_embeddings_file(embeddings_file_path: str) -> Iterable[str]:
        with open(embeddings_file_path, "r") as embeddings_file:
            words = []

            for idx, line in enumerate(embeddings_file):
                word_embedding_tuple = line.split("\t")
                word = word_embedding_tuple[0]
                words.append(word)
            return words

    @staticmethod
    def construct_word_idx_dict(vocab: Iterable[str]) -> Dict[str, int]:
        word_idx_dict = {}
        vocab_set = set(vocab)

        for idx, word in enumerate(vocab_set):
            word_idx_dict[word] = idx

        return word_idx_dict

    def idx_for_word(self, word: str) -> int:
        try:
            return self.word_idx_dict[word]
        except KeyError as e:
            return self.word_idx_dict["#UNK#"]

    def sentence_to_idx_tensor(self, sentence: List[str]) -> torch.LongTensor:
        return torch.LongTensor([self.idx_for_word(word) for word in sentence]).resize(len(sentence), 1)

    def forward(self, sentence: List[str]):
        # TODO: this needs to take a 2d IntTensor/LongTensor as input with dimensions (batch_size, padded_sentence_length)
        x = self.embedding_layer(self.sentence_to_idx_tensor(sentence))
        return x


def load_glove(path):
    """
    Load pretrained GloVe word embedding from specified path.

    Given a path this function open, reads and creates a dictionary that converts each word into 
    a Numpy array of the respective word embedding. Also extracts the size of these word embeddings
    to be use in filling in missing embeddings.

    Args:
        path: A path in the format of the string to the embedding file.

    Returns:
        A tuple of the word embedding dictionary and an integer size of each word embedding.
    """
    words, vectors, word2idx, idx = [], [], {}, 0
    with open(path) as f:
        for l in f:
            line = l.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx+=1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    
    glove = {w: vectors[word2idx[w]] for w in words}

    return glove, vectors[0].shape[0]

def embed(sentences, embedding_path): 
    """
    Embeds the provided list of sentences into the GloVe embedding space.

    Given a list of sentences, represented by an array of words, convert each word into their respective 
    Numpy array in the GloVe embedding space imported from the specfied path location. Replacing words that are missing 
    from the specified pretrained word embedding with a random Numpy array. Alternatively using all random Numpy arrays 
    if None is passed as the embedding_path.

    Args:
        sentences: A list where each item is sentence represented by a list of words.
        embedding_path: the path to the pretrained embedding file or None. Sets the embedding to random
        intiliastion if None is provided as the embedding_path.
    Returns:
        An embeddings list of lists for each sentence with a numpy array representing each word.
    """
    if embedding_path is not None:
        glove, emb_dim = load_glove(embedding_path)
    else:
        glove={}
        emb_dim=300
    embeddings = []

    for sentence in sentences:
        sentence_len = len(sentence)
        embedding = []
        for word in sentence:
            try:
                word_embeding = glove[word]
            except KeyError:
                word_embeding = np.random.normal(scale=0.6, size=(emb_dim,))
                glove[word] = word_embeding
            embedding.append(torch.as_tensor(word_embeding))
        torch_embedding = torch.cat(embedding, dim=0)
        # TODO Hyperparamater what to do with special tags specified in the tokenizer
        # Either a random array, array of zeros or use the word in the tag i.e. for "#date#" use "date"
        embeddings.append(torch.reshape(torch_embedding, (emb_dim, sentence_len)))

    return embeddings

    
