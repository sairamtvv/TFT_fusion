import argparse
import collections.abc
import datetime
import logging
import os
from typing import List, Union
import yaml


class ConfigParser:
    """Class to maintain config parameters.

    Enables config_nameparsing configuration parameters from a .yaml file and saving them into a
    dictionary. Also enables updating custom parameters of such configuration files.
    The path of the aimed configuration file can be specified, the default is 'config.yaml'.
    It contains a logger object for logging purposes.

    Attributes:
        logger: Logger with specified name.
    """

    def __init__(self, file_path: Union[None, str, List[str]] = None):
        self.logger = logging.getLogger(__name__)
        parser = argparse.ArgumentParser(
            description="For configuration", argument_default=argparse.SUPPRESS
        )
        if not file_path:
            parser.add_argument(
                "--ConfigFile",
                type=str,
                default="config.yaml",
                help="Path of the config file.",
            )
            known_args, _ = parser.parse_known_args()
            file_path_list: List[str] = [getattr(known_args, "ConfigFile")]

        elif isinstance(file_path, str):
            file_path_list: List[str] = [file_path]
        else:
            file_path_list: List[str] = file_path

        self._config_dict = dict()
        for file_path_entry in file_path_list:
            temp_config_dict = self._get_config(self.__class__.__name__, file_path_entry)
            self._config_dict = self.override_config(self._config_dict, temp_config_dict)

    def _get_config(self, module_name: str, config_file_path: str):
        """Gets the config parameters from a config file.

        Args:
            module_name:      The parameters are in module wise in config file
                              like mainParameters or dataEncodingParameters.
            config_file_path: Relative path to the config file. If None is given,
                              the default is "config.yaml".

        Returns:
            A dictionary which contains the different parameter types as keys and
            the actual parameters as values.
        """
        config_file_path: str = os.path.abspath(config_file_path)
        full_config_dict: dict
        self.logger.debug(f"Loading '{module_name}' config from '{config_file_path}'")
        with open(config_file_path, "r") as config_data:
            full_config_dict = yaml.load(config_data, Loader=yaml.SafeLoader)
        config_dict = full_config_dict[module_name]

        return config_dict

    def override_config(self, config_dict, override_dict):
        """Overrides configuration dictionary.

        Configuration parameter values are overridden with new parameter values
        which are specified in a given dictionary. It is possible to override only
        a subset of the parameter types.

        Args:
            config_dict:   Configuration dictionary which should be overridden.
            override_dict: Configuration dictionary which contains new parameter
                           values which should override the ones of the first
                           configuration dictionary.

        Returns: An updated configuration dictionary.
        """
        for k, v in override_dict.items():
            if isinstance(v, collections.abc.Mapping):
                config_dict[k] = self.override_config(config_dict.get(k, {}), v)
            else:
                config_dict[k] = v
        return config_dict

    def _get_date(self, date_str: str):
        """Gets a date format of a string, using the date format from config file.

        Args:
            date_str: String which shall be returned as a datetime object.

        Returns:
            A datetime object in a format which is specified in configuration file.
        """
        if date_str is None:
            return None
        else:
            return datetime.datetime.strptime(date_str, self._config_dict["date_format"])

    def get_custom_param(self, parameter_name):
        """Return value of given custom parameter."""
        if "custom_parameters" not in self._config_dict:
            self.logger.exception("No 'custom_parameters' in config.")
            raise ValueError("No 'custom_parameters' in config.")

        if parameter_name not in self._config_dict["custom_parameters"]:
            self.logger.exception(f"No '{parameter_name}' in 'custom_parameters' in config.")
            raise ValueError("No '{parameter_name}' in 'custom_parameters' in config.")

        return self._config_dict["custom_parameters"][parameter_name]


class BasePipelineConfig(ConfigParser):
    """Class to get prediction config parameters. Child

    Args:
        Inherited the main class.
    """

    @property
    def timeframes(self) -> dict:
        """All timeframes from the configuration dictionary."""

        supported_date_keys = ["start_date", "end_date", "historic_date", "reference_date"]

        return {
            timeframe_name: {
                key: (self._get_date(value) if key in supported_date_keys else value)
                for key, value in timeframe.items()
            }
            for timeframe_name, timeframe in self._config_dict["timeframes"].items()
        }

    @property
    def reference_date(self) -> datetime.datetime:
        """Reference date from the configuration dictionary."""
        return self._get_date(self._config_dict["reference_date"])

    @property
    def start_date(self) -> datetime.datetime:
        """Start date from the configuration dictionary."""
        return self._get_date(self._config_dict["start_date"])

    @property
    def historic_date(self) -> datetime.datetime:
        """Historic date from the configuration dictionary."""
        return self._get_date(self._config_dict["historic_date"])

    @property
    def end_date(self) -> datetime.datetime:
        """End date from the configuration dictionary."""
        return self._get_date(self._config_dict["end_date"])

    @property
    def region_list(self) -> list:
        """Regions for which time series and transaction data shall be considered."""
        return sorted(self._config_dict["regions"])

    @property
    def save_pickle(self) -> bool:
        """Boolean which indicates whether to save data as .pkl file temporarily."""
        return self._config_dict["pickle_save_pickle"]

    @property
    def hours_to_cache(self) -> int:
        """Number of hours that the temporarily saved data should be cached."""
        return int(self._config_dict["pickle_hours_to_cache"])

    @property
    def pickle_path(self) -> str:
        """Relative path to where .pkl files shall be saved."""
        return self._config_dict["pickle_path"]

    @property
    def quick_debug(self) -> bool:
        return self._config_dict["quick_debug"]

    @property
    def use_cached_ae(self) -> bool:
        return self._config_dict["use_cached_ae"]

    @property
    def use_processed_cached_dfs(self) -> bool:
        return self._config_dict["use_processed_cached_dfs"]

    @property
    def db_username(self) -> str:
        return self._config_dict["db_username"]

    @property
    def db_name(self) -> str:
        return self._config_dict["db_name"]


class TrainTestPipelineConfig(BasePipelineConfig):
    """Class to get main config parameters. Child

    Args:
        Inherited the main class.
    """

    @property
    def features(self) -> list:
        """All features from the configuration dictionary."""
        return sorted(self._config_dict["features"])

    @property
    def auxiliaries(self) -> list:
        """All auxiliaries from the configuration dictionary."""
        return sorted(self._config_dict["auxiliaries"])

    @property
    def target_variables(self) -> list:
        """All target variables from the configuration dictionary."""
        return sorted(self._config_dict["target_variables"])

    @property
    def object_level(self) -> str:
        """Level of the"""
        return self._config_dict["object_level"]

    @property
    def timeframe_type(self) -> str:
        """Date level on which th data shall be aggregated."""
        return self._config_dict["timeframe_type"]

    @property
    def load_all_variables(self) -> bool:
        return self._config_dict["load_all_variables"]

    # todo: to be deleted
    @property
    def feature_engineer(self) -> bool:
        return self._config_dict["feature_engineer"]

    @property
    def feature_engineering(self) -> dict:
        return {
            engineer_class: variables
            for engineer_class, variables in self._config_dict["feature_engineering"].items()
        }


class PredictPipelineConfig(BasePipelineConfig):
    """Class to get prediction config parameters. Child

    Args:
        Inherited the main class.
    """


class LoggerConfig(ConfigParser):
    """
       Class to generate logger config parameter
    Args:
        Inherited the main class
    """

    @property
    def log_directory(self) -> str:
        return self._config_dict["log_directory"]

    @property
    def log_name_prefix(self) -> str:
        return self._config_dict["log_name_prefix"]

    @property
    def log_level(self) -> str:
        return self._config_dict["log_level"]
