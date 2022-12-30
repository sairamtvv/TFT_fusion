import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pa_core.config import TrainTestPipelineConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting.models.baseline import Baseline
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

from pytorch_forecasting.data import GroupNormalizer,EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MQF2DistributionLoss, PoissonLoss, TweedieLoss,MAE,RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from torchmetrics import MeanAbsoluteError
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from pathlib import Path
import mlflow
import os



class TFTModelDefinition():

    def __init__(self, data, train_test_config):
        """Class holding multiple models """
        self.data = data
        self.train_test_config = train_test_config
        self.max_epochs = None
        for key, value in self.train_test_config.get_custom_param("tftmodeldefinition").items():
            setattr(self, key, value)



    def get_all_custom_params(self):
        return self.train_test_config._config_dict["custom_parameters"]

    def flatten_dict(self, dictionary, exclude=[], delimiter='_'):
        flat_dict = dict()
        for key, value in dictionary.items():
            if isinstance(value, dict) and key not in exclude:
                flatten_value_dict = self.flatten_dict(value, exclude, delimiter)
                for k, v in flatten_value_dict.items():
                    flat_dict[f"{key}{delimiter}{k}"] = v
            else:
                flat_dict[key] = value

        return flat_dict


    def log_dict_to_mlflow(self, _dict_to_log):
        for key, value in _dict_to_log.items():
            if key not in ["data", "train_test_config"]:
                mlflow.log_param(key, str(value))

    def log_all_parameters_to_mlflow(self, structure_data_obj, training, tft_model):
        #writing which parameters were considered static, time_varying etc.
        attributes = vars(structure_data_obj)
        # todo:Write baseline_output to file
        flat_dict = self.flatten_dict(self.get_all_custom_params(), exclude=["data", "train_test_config"])
        self.log_dict_to_mlflow(flat_dict)

        ic(attributes)
        flat_dict = self.flatten_dict(attributes, exclude=["data", "train_test_config"])

        self.log_dict_to_mlflow(flat_dict)

        # flat_dict = self.flatten_dict(training.get_parameters(), exclude=["data", "train_test_config"])
        # self.log_dict_to_mlflow(flat_dict)




    def create_tft_model(self, training):
        tft = TemporalFusionTransformer.from_dataset(
            training,
            # learning rate parameter below not meaningful for finding the learning rate but otherwise very important
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,  # most important hyperparameter apart from learning rate
            attention_head_size=self.attention_head_size,
            # number of attention heads. Set to up to 4 for large dataset
            dropout=self.dropout,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=self.hidden_continuous_size,  # set to <= hidden_size
            output_size=self.output_size,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=self.log_interval,
            # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience = self.reduce_on_plateau_patience,
            reduce_on_plateau_reduction=self.reduce_on_plateau_reduction,
            reduce_on_plateau_min_lr = self.reduce_on_plateau_min_lr,
            weight_decay=self.weight_decay,
            optimizer=self.optimizer,
            share_single_variable_networks=self.share_single_variable_networks
        )

        print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        return tft

    def get_params_for_runtype(self, lr_logger, early_stop_callback, logger, run_type):

        run_type = "quick_debug" if self.quick_debug else run_type

        def quick_debug_params():
            self.limit_train_batches,  self.callbacks, self.logger = 32,  [], False

        def find_lr_params():
            # How much of training dataset to check (float = fraction, int = num_batches).
            self.limit_train_batches, self.callbacks, self.logger = 1.0, [], False

        def train_params():
            self.limit_train_batches, self.callbacks, self.logger = 1.0, [lr_logger, early_stop_callback], logger
            # How much of training dataset to check (float = fraction, int = num_batches).
             # Default: ``1.0``.

        def default():
            print("Incorrect run_type")

        switcher = {
            "quick_debug": quick_debug_params,
            "find_lr": find_lr_params,
            "train": train_params
        }

        def switch(run_type):
            switcher.get(run_type, default)()

        switch(run_type)

        # if self.quick_debug:
        #     limit_train_batches = 32
        #     self.max_epochs = 3
        #     callbacks = []
        #     logger = False
        # else:
        #     if find_lr:
        #         limit_train_batches = 1.0  # How much of training dataset to check (float = fraction, int = num_batches).
        #         callbacks = []
        #         logger = False
        #     else:
        #         limit_train_batches = 1.0  # How much of training dataset to check (float = fraction, int = num_batches).
        #         callbacks = [lr_logger, early_stop_callback]  # Default: ``1.0``.



    def create_pl_trainer(self,lr_logger, early_stop_callback, logger, run_type):

        self.get_params_for_runtype(lr_logger, early_stop_callback, logger, run_type)
        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            enable_model_summary=True,
            gradient_clip_val=self.gradient_clip_val,

            limit_train_batches=self.limit_train_batches,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=self.callbacks,
            # callbacks=[lr_logger],

            logger=self.logger,
        )
        return trainer







    def configure_tft_network(self, training, run_type):
        early_stop_callback = EarlyStopping(monitor=self.early_stop_callback_monitor, min_delta=self.early_stop_callback_min_delta, patience=self.early_stop_callback_patience,
                                            verbose=True, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = self.configure_tensorboard_logger()
        trainer = self.create_pl_trainer(lr_logger, early_stop_callback, logger, run_type)
        tft_model = self.create_tft_model(training)
        return trainer, tft_model

    def configure_tensorboard_logger(self):
        # todo: Hard coded path in a way need to align with demandforecast.py
        # tensorboard_log_dir = Path(os.getcwd()) / "All_data" / self.train_test_config.get_custom_param("run_name")
        tensorboard_log_dir = f"{Path(os.getcwd())}/All_data/" + self.train_test_config.get_custom_param("run_name")

        logger = TensorBoardLogger(save_dir=tensorboard_log_dir)  # logging results to a tensorboard
        # version_no = logger.version
        # version_dir = f'{tensorboard_log_dir}/lightning_logs/version_{version_no}'
        # os.makedirs(version_dir, exist_ok=True)
        # mlflow.log_artifacts(version_dir)
        return logger


    


