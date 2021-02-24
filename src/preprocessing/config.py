from dataclasses import dataclass
from typing import List, Optional
from configparser import ConfigParser


class MissingConfigurationParam(Exception):
    pass


class ConfigurationTypeMismatch(Exception):
    pass


@dataclass
class Config:
    # TODO: add more config params as required and parse them
    epochs: int
    early_stopping: int
    lr: float
    path_eval_result: Optional[str]
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

        try:
            return Config(int(config["epochs"]), int(config["early_stopping"]), float(config["lr"]),
                          config.get("path_eval_result"), ensemble_configs)
        except KeyError as e:
            raise MissingConfigurationParam(e)
        except TypeError as e:
            raise ConfigurationTypeMismatch(e)
