from torch.utils.data import Dataset
from torch.nn.functional import pad
from sentence_classifier.preprocessing.reader import load
from sentence_classifier.preprocessing.tokenisation.tokeniser import parse_tokens
import torch

def map_dict(path):
    word2idx, idx = {}, 1
    with open(path) as f:
        for l in f:
            line = l.split()
            word = line[0]
            word2idx[word] = idx
            idx+=1
    
    return word2idx

class DatasetQuestions(Dataset):
    """
    This extended class of Dataset facilitates the work of DataLoader for managing (eg. batching) the questions dataset.
    """

    def __init__(self, filepath, tokenisation_rules, dict_path):
        self.questions, self.classifications = load(filepath)

        # Map questions to tokenised questions
        self.tokenised_questions = list(map(lambda x: parse_tokens(x, tokenisation_rules), self.questions))

        self.embedding_map = map_dict(dict_path)

        self.longest_sequence = 0

    def __len__(self):
        return len(self.tokenised_questions)

    def __getitem__(self, index: int):
        return self.tokenised_questions[index], self.classifications[index]
        # return self.transform(self.embedding[index]), self.classifications[index]
    
    def transform(self, question):
        dim = len(question)
        mapped_question = []
        for w in question:
            try:
                mapped_question.append(self.embedding_map[w])
            except KeyError:
                mapped_question.append(self.embedding_map["#UNK#"])   
        return pad(input=torch.LongTensor(mapped_question), pad=(0, self.longest_sequence-dim), mode='constant', value=0)

    # this method is passed to DataLoader class for making the size of the sequences in a batch consistent
    def collate_fn(self, batch):
        self.longest_sequence = 0
        # save the max length of the sequences in the batch
        for q, l in batch:
            self.longest_sequence = max(self.longest_sequence, len(q))
        qs, ls = [], []
        # modify the batch by padding the sequences to match the size of the longest sequence
        for q, l in batch:
            qs.append(self.transform(q))
            ls.append(l)
        return (qs, ls)