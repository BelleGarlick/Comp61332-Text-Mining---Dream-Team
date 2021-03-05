from unittest import TestCase
from sentence_classifier.preprocessing.config import Config, MissingConfigurationParam
import os
import shutil


class ConfigParserTest(TestCase):

    def create_mock_config_file(self, bad=False) -> str:
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
                "path_eval_result = /somedir/eval_out.txt\n"
            ])

            return mock_config_file.name

    def test_read_config(self):
        mock_config_filepath = self.create_mock_config_file()
        config = Config.from_config_file(mock_config_filepath)

    def test_read_bad_config(self):
        mock_config_filepath = self.create_mock_config_file(bad=True)
        self.assertRaises(MissingConfigurationParam, lambda: Config.from_config_file(mock_config_filepath))

    def tearDown(self):
        shutil.rmtree("testfiles")