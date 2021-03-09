import argparse
import sys

from torch.utils import data

from sentence_classifier.utils.config import Config
from sentence_classifier.utils.one_hot_encoding import OneHotEncoder

from sentence_classifier.preprocessing.dataloading import DatasetQuestions
from torch.utils.data import DataLoader

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


if __name__ == "__main__":
    #Command line arguments parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run the code to train a model')
    parser.add_argument('--test', action='store_true', help='Run the code to test a model')
    parser.add_argument('--config', nargs='?', default='../data/config.ini')
    args = parser.parse_args(sys.argv[1:])
    config = Config.from_config_file(args.config)

    # TODO: Handle the case when the argument is --test instead of --train
    # Load the dataset
    train_dataset = DatasetQuestions(config.path_train, tokenisation_rules=tokenisation_rules)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=train_dataset.collate_fn)

    print(train_loader)

    for emb_data_point in train_loader:
        print(emb_data_point)

    print(train_dataset.longest_sequence)

    # one_hot_encoder = OneHotEncoder()
    # one_hot_encoding = one_hot_encoder.encode(tokenised_questions, update_corpus=True)
