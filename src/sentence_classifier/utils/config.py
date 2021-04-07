from dataclasses import dataclass
from typing import List, Optional
from typing import Literal  # TODO: requires python3.8+
from configparser import ConfigParser
from sentence_classifier.models.model import Model


class MissingConfigurationParam(Exception):
    pass


class ConfigurationTypeMismatch(Exception):
    pass


class ConfigurationException(Exception):
    pass


@dataclass
class Config:
    # TODO: add more config params as required and parse them
    path_train: str
    path_test: str
    epochs: int
    early_stopping: int
    lr: float
    path_eval_result: Optional[str]

    word_embeddings: Literal["random", "glove"]  # TODO: requires python3.8+, remove if Kilburn VMs don't support it
    tune_word_embeddings: Literal["freeze", "tune"]
    path_word_embeddings: Optional[str]
    word_embedding_dim: Optional[int]

    sentence_embedder: Literal["bow", "bilstm"]  # TODO: requires python3.8+
    bilstm_input_dim: Optional[int]
    bilstm_hidden_dim: Optional[int]

    classifier_input_dim: int
    ensemble_configs: List['Config']

    @staticmethod
    def from_config_file(filepath: str) -> 'Config':
        config_parser = ConfigParser()
        config_parser.sections()

        config_parser.read(filepath)
        config = config_parser["main"]

        ensemble_config_keys = filter(lambda config_key: config_key.startswith("ensemble_config"), config.keys())
        ensemble_config_paths = [config[ensemble_config_key] for ensemble_config_key in ensemble_config_keys]
        ensemble_configs = [Config.from_config_file(ensemble_config_path) for ensemble_config_path in ensemble_config_paths]

        word_embeddings = Config.parse_word_embedding_config(config.get("word_embeddings"))

        if word_embeddings == "glove" and config.get("path_word_embeddings") is None:
            raise MissingConfigurationParam('path_word_embeddings must be set when word_embeddings is set to glove')
        elif word_embeddings == "glove" and config.get("word_embedding_dim") is None:
            raise MissingConfigurationParam('word_embedding_dim must be set when word_embeddings is set to random')
        else:
            pass

        train_word_embeddings = Config.parse_train_word_embedding_config(config["train_word_embeddings"])

        sentencer_embedder = Config.parse_sentence_embedder_config(config.get("sentence_embedder"))
        if sentencer_embedder == "bilstm":
            if config.get("bilstm_input_dim") is None:
                raise MissingConfigurationParam('bilstm_input_dim must be set when sentencer_embedder is set to bilstm')
            if config.get("bilstm_hidden_dim") is None:
                raise MissingConfigurationParam('bilstm_hidden_dim must be set when sentencer_embedder is set to bilstm')
            else:
                pass
        else:
            pass

        try:
            return Config(config["path_train"],
                          config["path_test"],
                          int(config["epochs"]),
                          int(config["early_stopping"]),
                          float(config["lr"]),
                          config.get("path_eval_result"),
                          word_embeddings,
                          train_word_embeddings,
                          config.get("path_word_embeddings"),
                          int(config.get("word_embedding_dim")),
                          sentencer_embedder,
                          int(config.get("bilstm_input_dim")),
                          int(config.get("bilstm_hidden_dim")),
                          int(config["classifier_input_dim"]),
                          ensemble_configs)
        except KeyError as e:
            raise MissingConfigurationParam(e)
        except TypeError as e:
            raise ConfigurationTypeMismatch(e)

    @staticmethod
    def build_model_from_config(filepath: str) -> Model:
        config = Config.from_config_file(filepath)
        model_builder = Model.Builder()
        fine_tune = config.tune_word_embeddings == "tune"
        freeze = not fine_tune

        if config.word_embeddings == "glove":
            model_builder.with_glove_word_embeddings(config.path_word_embeddings, freeze=freeze)
        else:
            model_builder.with_random_word_embeddings(config.path_train, config.word_embedding_dim, freeze=freeze)

        if config.sentence_embedder == "bow":
            model_builder.with_bow_sentence_embedder()
        else:
            model_builder.with_bilstm_sentence_embedder(config.bilstm_input_dim, config.bilstm_hidden_dim)

        model_builder.with_classifier(config.classifier_input_dim)

        model = model_builder.build()
        return model

    @staticmethod
    def parse_word_embedding_config(word_embedding_config_str: str) -> Literal["random", "glove"]:
        if word_embedding_config_str == "random":
            return "random"
        elif word_embedding_config_str == "glove":
            return "glove"
        else:
            raise ConfigurationException(f'word_embedding must be "glove" or "random"')

    @staticmethod
    def parse_train_word_embedding_config(train_word_embedding_config_str: str) -> Literal["freeze", "tune"]:
        if train_word_embedding_config_str == "freeze":
            return "freeze"
        elif train_word_embedding_config_str == "tune":
            return "tune"
        else:
            raise ConfigurationException(f'train_word_embedding must be "freeze" or "tune"')

    @staticmethod
    def parse_sentence_embedder_config(sentence_embedding_config_str: str) -> Literal["bow", "bilstm"]:
        if sentence_embedding_config_str == "bow":
            return "bow"
        elif sentence_embedding_config_str == "bilstm":
            return "bilstm"
        else:
            raise ConfigurationException(f'sentence_embedder must be "bow" or "bilstm"')
