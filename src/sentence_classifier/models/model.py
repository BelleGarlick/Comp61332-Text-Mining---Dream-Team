from torch import nn

from typing import Optional, Union


from sentence_classifier.models.embedding import WordEmbeddings
from sentence_classifier.models.bagofwords import BagOfWords
from sentence_classifier.models.BiLSTM import BiLSTM
from sentence_classifier.models.classifier_nn import ClassifierNN
from sentence_classifier.utils.vocab import VocabUtils

SentenceEmbedder = Union[BagOfWords, BiLSTM]


class ModelBuildError(Exception):
    pass


class DimensionMismatchException(Exception):
    pass


class Model(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, sentence_embeddings: SentenceEmbedder, classifier: ClassifierNN):
        super(Model, self).__init__()

        self.word_embeddings = word_embeddings
        self.sentence_embeddings = sentence_embeddings
        self.classifier = classifier

    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.sentence_embeddings(x)
        x = self.classifier(x)

        return x

    class Builder:
        def __init__(self):
            self.word_embeddings: Optional[WordEmbeddings] = None
            self.sentence_embeddings: Optional[SentenceEmbedder] = None
            self.classifer: Optional[ClassifierNN] = None

        def with_glove_word_embeddings(self, embeddings_file_path: str) -> 'Model.Builder':
            word_embeddings = WordEmbeddings.from_embeddings_file(embeddings_file_path)
            self.word_embeddings = word_embeddings
            return self

        def with_random_word_embeddings(self, training_data_file: str, emb_dim) -> 'Model.Builder':
            """
            Uses the tokenized training data as a vocabulary
            """
            vocab = VocabUtils.vocab_from_training_data(training_data_file)
            word_embeddings = WordEmbeddings.from_random_embedding(vocab, emb_dim)
            self.word_embeddings = word_embeddings
            return self

        def with_bow_sentence_embedder(self) -> 'Model.Builder':
            bow = BagOfWords()
            self.sentence_embeddings = bow
            return self

        def with_bilstm_sentence_classifier(self, emb_dim, hidden_dim) -> 'Model.Builder':
            bilstm = BiLSTM(emb_dim, hidden_dim)
            self.sentence_embeddings = bilstm
            return self

        def with_classifier(self, classifier_input_dim: int):
            classifier_nn = ClassifierNN(classifier_input_dim)
            self.classifer = classifier_nn
            return self

        def build(self) -> 'Model':
            if self.word_embeddings is None:
                raise ModelBuildError("Need to set word_embeddings layer for the model")
            elif self.sentence_embeddings is None:
                raise ModelBuildError("Need to set sentence_embeddings layer for the model")
            elif self.classifer is None:
                raise ModelBuildError("Need to set classifier layer for the model")
            else:
                self.check_word_embedding_sentence_embedding_dim_match()
                self.check_sentence_embedder_classifier_input_dim_match()

                model = Model(self.word_embeddings, self.sentence_embeddings, self.classifer)
                return model

        def check_word_embedding_sentence_embedding_dim_match(self) -> None:
            if not isinstance(self.sentence_embeddings, BiLSTM):
                return
            else:
                sentence_embedder_input_dim = self.sentence_embeddings.lstm.input_size
                word_embedding_dim = self.word_embeddings.embedding_layer.embedding_dim
                if word_embedding_dim != sentence_embedder_input_dim:
                    raise DimensionMismatchException(f'Mismatch between the BiLSTM input dim ({sentence_embedder_input_dim}) and the '
                                                     f'word-embeddings dim ({word_embedding_dim})')
                else:
                    return

        def check_sentence_embedder_classifier_input_dim_match(self) -> None:
            sentence_emedding_dim = self.sentence_embeddings.output_dim if isinstance(self.sentence_embeddings, BiLSTM) else self.word_embeddings.embedding_layer.embedding_dim
            classifier_input_dim = self.classifer.input_dim
            if sentence_emedding_dim != classifier_input_dim:
                raise DimensionMismatchException(f'Mismatch between the Classifier input dim ({classifier_input_dim}) '
                                                 f'and the sentence embedding output dim ({sentence_emedding_dim})')
            else:
                return
