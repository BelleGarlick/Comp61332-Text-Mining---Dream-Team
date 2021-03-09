from torch.utils.data import Dataset
from torch.nn.functional import pad
from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation.tokeniser import parse_tokens
from sentence_classifier.preprocessing.embedding import embed

class DatasetQuestions(Dataset):
    def __init__(self, filepath, tokenisation_rules):
        self.questions, self.classifications = load(filepath)

        # Map questions to tokenised questions
        self.tokenised_questions = list(map(lambda x: parse_tokens(x, tokenisation_rules), self.questions))
        
        #Using the pretrained GloVe embedding
        self.embedding = embed(self.tokenised_questions, "../data/glove.small.txt")

        #Using the randomally intilised embeddings
        self.rand_embedding = embed(self.questions, None)

        self.longest_sequence = 20
    def collate_fn(self, batch):
        self.longest_sequence = 0
        for q, l in batch:
            self.longest_sequence = max(self.longest_sequence, len(q[0]))
        new_batch = []
        for q, l in batch:
            new_batch.append((self.transform(q), l))
        return new_batch

    def transform(self, qembedded):
        dim = len(qembedded[0])
        return pad(input=qembedded, pad=(0, self.longest_sequence-dim), mode='constant', value=0)

    def __len__(self):
        return len(self.tokenised_questions)

    def __getitem__(self, index: int):
        return self.embedding[index], self.classifications[index]
        # return self.transform(self.embedding[index]), self.classifications[index]