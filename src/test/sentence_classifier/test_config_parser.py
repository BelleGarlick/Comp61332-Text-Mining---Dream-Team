from unittest import TestCase
from sentence_classifier.utils.config import Config, MissingConfigurationParam, ConfigurationException
import os
import shutil


class ConfigParserTest(TestCase):

    def create_mock_config_file(self, bad=False, bad_word_embedding=False, bad_sentence_embedder=False) -> str:
        if not os.path.exists("testfiles"):
            os.mkdir("testfiles")

        with open("testfiles/mock-config.ini", "w") as mock_config_file:
            mock_config_file.writelines([
                "[main]\n",
                "path_train = ../data/train.txt\n",
                "path_test = ../data/test.txt\n",
                "epochs = 10\n" if not bad else "",
                "lr = 0.005\n",
                "early_stopping = 20\n",
                "word_embeddings = glove\n" if not bad_word_embedding else "",
                "path_word_embeddings =../ data / glove.small.txt\n",
                "word_embedding_dim = 300\n",
                "sentence_embedder = bow\n" if not bad_sentence_embedder else "",
                "bilstm_input_dim = 300\n",
                "bilstm_hidden_dim = 300\n",
                "classifier_input_dim = 300\n",
                "path_eval_result = /somedir/eval_out.txt\n"
            ])

            return mock_config_file.name

    def test_read_config(self):
        mock_config_filepath = self.create_mock_config_file()
        config = Config.from_config_file(mock_config_filepath)

    def test_read_bad_config(self):
        mock_config_filepath = self.create_mock_config_file(bad=True)
        self.assertRaises(MissingConfigurationParam, lambda: Config.from_config_file(mock_config_filepath))

    def test_invalid_word_embedding_specified(self):
        mock_config_filepath = self.create_mock_config_file(bad_word_embedding=True)
        self.assertRaises(ConfigurationException, lambda: Config.from_config_file(mock_config_filepath))

    def test_invalid_sentence_embedder_specified(self):
        mock_config_filepath = self.create_mock_config_file(bad_sentence_embedder=True)
        self.assertRaises(ConfigurationException, lambda: Config.from_config_file(mock_config_filepath))

    def tearDown(self):
        shutil.rmtree("testfiles")
