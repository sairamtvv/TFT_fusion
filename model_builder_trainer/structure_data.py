import itertools
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting.models.baseline import Baseline
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

from pytorch_forecasting.data import GroupNormalizer,EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MQF2DistributionLoss, PoissonLoss, TweedieLoss,MAE,RMSE
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from torchmetrics import MeanAbsoluteError
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from pa_core.config import TrainTestPipelineConfig

#todo:Productline averages to be added along with group normalization

class StructuringDataset:

    def __init__(self, data, train_test_config):

        self.data = data
        self.train_test_config = train_test_config
        self.static_categoricals = self.get_static_categoricals()
        self.time_varying_known_categoricals = self.get_time_varying_known_categoricals()
        self.time_varying_unknown_reals = self.get_time_varying_unknown_reals()
        self.training = None

        for key, value in self.train_test_config.get_custom_param("structuringdataset").items():
            setattr(self, key, value)



    def none_return_fun(self):
        return None

    def function_parser(self, option):

        # this doesn't have ()
        funct_parser = {
            "nanlabelencoder": NaNLabelEncoder, "robust": RobustScaler,
            "minmax": MinMaxScaler, "standard": StandardScaler, "none": self.none_return_fun
        }

        return funct_parser[option]

    def dont_include(self):

        ['SYSTEM', 'index', 'NEW_RESEARCHERS', 'CUM_NO_RESEARCHERS', 'YEAR',
         ]

        dont_include = ["index", "New_researchers", "Cumulative_number_of_researchers", "TIMEFRAME_ID","FIRST_DAY_OF_TIMEFRAME"] + \
                       [ "LAST_DAY_OF_TIMEFRAME", "OBJECT_ID", "REGION_ALIAS" ]
        because_null = ["INPL"]

    def get_static_reals(self):
        static_reals = []
        return static_reals

    def get_time_varying_known_reals(self):
        #todo: check W_CAMPAIGN_PR and W_CAMPAIGN_CT are they reals are category
        #todo: Add style_id as category
        time_varying_known_reals = ['DIFUSION_COEFFICIENT', 'SCREENING_POTENTIAL', 'REPRODUCIBILITY'] + \
                                   ['DISCREPANCY', 'MEASUREMENT_FLAW']
        return time_varying_known_reals

    def get_time_varying_known_categoricals(self):
        #todo:Do you want to add style id as an  category
        #todo: Add null_info null_cat for category as columns to model_builder_trainer
        #todo: Check if we want to add the cyclical encoding
        time_varying_known_categoricals = ['SYSTEM']

        return time_varying_known_categoricals



    def get_static_categoricals(self):
        #todo: Add style here
        static_categoricals = []
        return static_categoricals




    def get_time_varying_unknown_reals(self):
        time_varying_unknown_reals = ['YES', 'NO', 'Y_AVG', 'X_NORM', 'YT_AVERAGE', 'Z_FACTOR',
         'SUM_OF_SQUARES', 'SS_TOTAL',  'DOF_AVG', 'log_ANNOVA_NORM', 'avg_ANNOVA_NORM_by_SYSTEM',
         'ANNOVA_NORM_lagged_1', 'ANNOVA_NORM_lagged_2']

        return time_varying_unknown_reals

    # todo: Check if there are nans in any  numerical columns being passed to the model

    def typecast(self):
        #todo: Shift this typecast into tft  demandforecast.py so that when it is typecasted, it does the way I would want
        #todo: Why is my typecasting different than already present. Give it a thought
        ic("typecasting is known to take some time")
        self.data[self.get_static_categoricals()] = self.data[self.get_static_categoricals()].astype(str).astype('category')
        self.data[self.get_static_reals()] = self.data[self.get_static_reals()].astype(float).astype("float32")
        self.data[self.get_time_varying_known_categoricals()] = self.data[self.get_time_varying_known_categoricals()].astype(str).astype('category')
        self.data[self.get_time_varying_known_reals()] = self.data[self.get_time_varying_known_reals()].astype(float).astype("float32")
        self.data[self.get_time_varying_unknown_reals()] = self.data[self.get_time_varying_unknown_reals()].astype(float).astype("float32")

        self.data["ANNOVA_NORM"] = self.data["ANNOVA_NORM"].astype(float).astype("float32")
        self.data['time_idx'] = self.data['time_idx'].apply(np.int64)
        assert(self.data["time_idx"].dtype.kind == "i"), "time_idx must be of type integer (i)"

    def rename_columns(self):
       pass






    def categorical_nan_encoder(self):
        """Making a dictionary for the categorical nan encoder"""
        categories = self.get_static_categoricals() + self.get_time_varying_known_categoricals()
        encoder = {}
        # encoder['__group_id__transaction'] = NaNLabelEncoder(add_nan=True, warn=True)
        # encoder['__group_id__ITEM_ID'] = NaNLabelEncoder(add_nan=True, warn=True)
        for i in categories:
            encoder[i] = self.function_parser(self.categorical_encoding)(add_nan=True, warn=True)
        return encoder



    def numerical_scalar(self):
        #todo:this can be made elegant by defining a none trturning function
        scaler = {}
        # for i in reals_cyclical:
        #     scaler[i] = None
        #reals = reals_known_fut + reals_unknown + static_type_reals

        for i in self.get_time_varying_known_reals():
            scaler[i] = self.function_parser(self.time_varying_known_numerical_scaler)()

        for i in self.get_time_varying_unknown_reals():
            scaler[i] = self.function_parser(self.time_varying_unknown_numerical_scaler)()

        for i in self.get_static_reals():
            scaler[i] = self.function_parser(self.static_numerical_scaler)()

        return scaler

    def structure_as_timeseries_dataset(self):
        #todo: Expected an error and need to comment out dome scaler or relative_idx

        self.typecast()
        self.rename_columns()


        max_prediction_length = self.max_prediction_length
        max_encoder_length = self.max_encoder_length
        training_cutoff = self.data["time_idx"].max() - self.max_prediction_length

        if len(self.group_ids) > 1:
            target_normalizer = GroupNormalizer(groups=self.group_ids, transformation="softplus")
        else:
            target_normalizer = EncoderNormalizer(method="robust")



        training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="ANNOVA_NORM",
            group_ids=["SYSTEM"],
            #weight="weight",
            min_encoder_length=self.min_encoder_length,
            # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            # min_prediction_idx = training_cutoff + 1,
            max_prediction_length=max_prediction_length,
            # lags= {'DEMAND_PCS':[1,2] },
            static_categoricals=self.get_static_categoricals(),
            static_reals=self.get_static_reals(),
            # + list(extracted_features.columns)[:-2],
            time_varying_known_categoricals=self.get_time_varying_known_categoricals(),
            # variable_groups=groups,  # group of categorical variables can be treated as one variable
            time_varying_known_reals=["time_idx"] + self.get_time_varying_known_reals(),
            # time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                                           'ANNOVA_NORM',
                                       ] + self.get_time_varying_unknown_reals(),
            allow_missing_timesteps=True,

            target_normalizer=target_normalizer,

            # GroupNormalizer(groups=["ITEM_ID"],transformation= "softplus"),

            # GroupNormalizer(groups=["PRODUCTLINE","ITEM_ID"], center=False, transformation=dict(forward=torch.log1p)),

            #

            #

            # EncoderNormalizer(transformation=dict(forward=torch.log1p)),

            # GroupNormalizer(groups=["PRODUCTLINE", "ITEM_ID"],transformation= "softplus")],

            # EncoderNormalizer(method = "robust"),

            # MultiNormalizer(normalizers: [GroupNormalizer(groups=["PRODUCTLINE", "ITEM_ID"]),]
            #
            # use softplus and normalize by group
            add_relative_time_idx=True,
            #add_target_scales=True,
            add_encoder_length=True,
            randomize_length=True,
            categorical_encoders=self.categorical_nan_encoder(),
            scalers=self.numerical_scalar()

        )
        return training, training_cutoff



    def get_training_parameters(self, training):
        #todo: For every run training parameters, hyperparameters
        #todo: should be printed into file to refer what parameters were sent to the model_executor
        return training.get_parameters()


    def prepare_dataloader(self,training, training_cutoff, data, predict=False):

        validation = TimeSeriesDataSet.from_dataset(training, data, predict=predict,
                                                    stop_randomization=True,
                                                    min_prediction_idx=training_cutoff + 1)
        train_dataloader = training.to_dataloader(train=True, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)

        val_dataloader = validation.to_dataloader(train=False, batch_size=self.batch_size * 10,
                                                  num_workers=self.num_workers)
        return train_dataloader, val_dataloader











    #Functions unused currently

    def cyclical_encodings(self):
        # Reference Month - 1 to 12
        self.data['rm_sin'] = np.sin(2 * np.pi * pd.to_numeric(self.data['REFERENCE_MONTH']) / pd.to_numeric(self.data['REFERENCE_MONTH']).max())
        self.data['rm_cos'] = np.cos(2 * np.pi * pd.to_numeric(self.data['REFERENCE_MONTH']) / pd.to_numeric(self.data['REFERENCE_MONTH']).max())

        # Reference Week - 1 to 52
        self.data['rw_sin'] = np.sin(2 * np.pi * pd.to_numeric(self.data['REFERENCE_WEEK']) / pd.to_numeric(self.data['REFERENCE_WEEK']).max())
        self.data['rw_cos'] = np.cos(2 * np.pi * pd.to_numeric(self.data['REFERENCE_WEEK']) / pd.to_numeric(self.data['REFERENCE_WEEK']).max())

        #Week of month - 1 to 4
        self.data['wom_sin'] = np.sin(2 * np.pi * pd.to_numeric(self.data['WEEK_OF_MONTH']) / pd.to_numeric(self.data['WEEK_OF_MONTH']).max())
        self.data['wom_cos'] = np.cos(2 * np.pi * pd.to_numeric(self.data['WEEK_OF_MONTH']) / pd.to_numeric(self.data['WEEK_OF_MONTH']).max())

    def get_varible_groups(self):
        self.data.DEPT_NULL_INFO = self.data.DEPT_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")
        self.data.EK_NULL_INFO = self.data.EK_NULL_INFO.apply(lambda x: "1" if x == 'null_cat' else "0").astype(
            "category")
        self.data.PRODUCTLINE_NULL_INFO = self.data.PRODUCTLINE_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")
        self.data.GENDER_NULL_INFO = self.data.GENDER_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")
        self.data.PRODUCTLINE_NULL_INFO = self.data.PRODUCTLINE_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")
        self.data.SEASONALITY_NULL_INFO = self.data.SEASONALITY_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")
        self.data.SHOPASS_NULL_INFO = self.data.SHOPASS_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")
        self.data.U_PRODUCTLINE_NULL_INFO = self.data.U_PRODUCTLINE_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")
        self.data.AGE_DETAIL_NULL_INFO = self.data.AGE_DETAIL_NULL_INFO.apply(
            lambda x: "1" if x == 'null_cat' else "0").astype("category")

        variable_groups = {
            "DEPT": ["DEPT", "DEPT_NULL_INFO"],
            "EK": ["EK", "EK_NULL_INFO"],
            "GENDER": ["GENDER", "GENDER_NULL_INFO"],
            "PRODUCTLINE": ["PRODUCTLINE", "PRODUCTLINE_NULL_INFO"],
            "SEASONALITY": ["SEASONALITY", "SEASONALITY_NULL_INFO", ],
            "SHOPASS": ["SHOPASS", "SHOPASS_NULL_INFO"],
            "U_PRODUCTLINE": ["U_PRODUCTLINE", "U_PRODUCTLINE_NULL_INFO"],
            "AGE_DETAIL": ["AGE_DETAIL", "AGE_DETAIL_NULL_INFO"]

        }
        variable_groups_lst = list(itertools.chain.from_iterable(variable_groups.values()))

        return variable_groups


