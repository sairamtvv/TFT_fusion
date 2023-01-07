import os

import pa_core
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pa_core
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch


from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MQF2DistributionLoss, \
    PoissonLoss, TweedieLoss, MAE, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from torchmetrics import MeanAbsoluteError
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from pathlib import Path
from pa_demand_forecast.raw_forecast_tft.model_builder_trainer.structure_data import \
    StructuringDataset
import mlflow


from pa_demand_forecast.raw_forecast_tft.predict.interpret_tft_postpredict import \
    InterpretTFTPostPredict
from pa_demand_forecast.raw_forecast_tft.predict.tft_evaluate import EvaluateTFT
import logging
logger = logging.getLogger(__name__)

class Predict:
    # todo: plot the actual histogram of demand_pot over the tft_histogram of demand_pot

    def __init__(self, data, train_test_config):
        self.data = data
        self.train_test_config = train_test_config

        # todo: can make this elegant
        # these custom parameters needs to be present in both train and test config. But only predict config is used
        for key, value in self.train_test_config.get_custom_param("predict_class").items():
            setattr(self, key, value)
        self.update_params_from_config()


    def update_params_from_config(self):
        self.max_prediction_length = self.train_test_config.get_custom_param("structuringdataset")[
            "max_prediction_length"]
        run_name = self.train_test_config.get_custom_param("run_name")
        path_tomake_dir_fig = f"{Path.cwd()}/All_data/{run_name}"
        Path(path_tomake_dir_fig + "/images").mkdir(exist_ok=True)
        self.images_dir = str(Path(os.path.join(path_tomake_dir_fig, "images")))
        self.is_real_prediction = self.train_test_config.get_custom_param("is_real_prediction")
        self.run_pipeline_from_dataframe = self.train_test_config.get_custom_param("run_pipeline_from_dataframe")


    def prepare_data(self, predict):
        structure_data_obj = StructuringDataset(self.data, self.train_test_config)
        training, training_cutoff = structure_data_obj.structure_as_timeseries_dataset()
        train_dataloader, val_dataloader = structure_data_obj.prepare_dataloader(training,
                                                                                 training_cutoff,
                                                                                 self.data,
                                                                                 predict=predict)
        return structure_data_obj, training, training_cutoff, train_dataloader, val_dataloader



    def predict(self, best_model_path, fcst_tmfrm_nm=None):
        structure_data_obj, training, training_cutoff, train_dataloader, \
            val_dataloader = self.prepare_data(predict=True)

        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        new_pred, new_x, new_index = best_tft.predict(val_dataloader, mode="prediction",
                                                      return_x=True, return_index=True)

        new_pred_raw, new_x_raw, new_index_raw = best_tft.predict(val_dataloader, mode="raw",
                                                                  return_x=True, return_index=True)

        # todo: Add quantiles to the result_df
        # todo: Making the actual train_df and merging with the incoming test data
        # todo: Adding individual plots making ANN as a input parameter
        # todo: Doing the correct preprocessing for the target data
        #todo: Write the code for data not in validation data set
        #todo: actuals_vs_predictions plot, make a directory in images and save
        #todo: currently time_idx has a standard scalar which is very wrong, remove it


        result_tft_df = self.make_tft_predictions_df(best_tft, val_dataloader, new_pred, new_x, new_index)
        result_tft_df = self.get_desired_df_for_gross_forecast(result_tft_df)


        # interpretation
        interpret_tft_obj = InterpretTFTPostPredict(self.data, self.train_test_config)
        interpret_tft_obj.interpret_model(best_tft, new_pred_raw)
        #interpret_tft_obj.partial_dependency(best_tft, val_dataloader, lst_input_feat=None)



        # Evaluation and comparision with the other models
        if fcst_tmfrm_nm is None:
            fcst_tmfrm_nm = "validation"

        #todo: check if self.data contains train+test data by now
        evaluatetft_obj = EvaluateTFT(self.data, self.train_test_config)
        rslts_dict = evaluatetft_obj.evaluate(fcst_tmfrm_nm, best_model_path, result_tft_df)
        evaluatetft_obj.visualize_individual_items(rslts_dict, best_tft, val_dataloader, new_pred_raw, new_x_raw,
                                   new_index_raw, new_index, new_x, new_pred)

        evaluatetft_obj.qqplot(result_tft_df)




        evaluatetft_obj.log_all_prediction_to_mlflow(best_tft, val_dataloader, best_model_path)

        return result_tft_df

    def predict_itemids_notin_validationset(self):
        pass

    def remove_itemids_with_all_quantiles_nan(self):
        pass













    def get_df_for_forecast(self, best_model_path):
        """
        Extract only those ITEM_IDS that are present in the trained model
        and filter the train+prediction data based on those ITEM_IDS alone.
        """
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        # train_df = pd.read_pickle(logged_train_data_path, compression="zip")
        #
        # self.data = pd.concat([train_df, self.data], axis=0)

        # #todo: Check if dropping duplicates is required

        dict_trained_item_ids = dict(best_tft.hparams)["embedding_labels"]["ITEM_ID"]
        dict_trained_item_ids.pop("nan")
        trained_item_ids_df = pd.DataFrame(list(dict_trained_item_ids.items()), columns=['ITEM_ID','item_id_mapping'])
        self.data["ITEM_ID"] = self.data["ITEM_ID"].astype(str)
        self.data = pd.merge(trained_item_ids_df, self.data, how="inner", on=["ITEM_ID"])

        assert len(self.data) > 0, "The trained item_ids and the prediction item_ids are different and no common ids among them"



    def predict_evaluate_with_new_data(self, fcst_tmfrm_nm=None):
        """Read the old data and add the new incoming data """
        # read the old train_data from the data frame



        # Using Mlflow to load the best_model_path and train_data path saved inside the MLFLOW experiment

        logged_checkpoint_path = f"./mlruns/{self.toread_mlflow_experiment_id}/" \
                            f"{self.toread_mlflow_run_id}/artifacts/lightning_logs/version_0/checkpoints/" \
                            f"{self.toread_mlflow_ckpt_filename}"

        # logged_train_data_path = f"./mlruns/{self.toread_mlflow_experiment_id}/" \
        #                     f"{self.toread_mlflow_run_id}/artifacts/" \
        #                     f"{self.toread_mlflow_train_filename}"

        self.get_df_for_forecast(logged_checkpoint_path)


        # #make a outer join with the new data
        # #self.data = pd.merge(old_train_df, self.data, how="outer", on=["time_idx", "ITEM_ID"])
        # self.data = self.get_only_train_data_from_df()
        result_df = self.predict(logged_checkpoint_path, fcst_tmfrm_nm)



        return result_df

    def get_desired_df_for_gross_forecast(self, result_df):

        self.data["time_idx"] = (round(((self.data["FIRST_DAY_OF_TIMEFRAME"] - self.data[
            "FIRST_DAY_OF_TIMEFRAME"].min()).dt.days) / 7)).astype(int)
        self.data["ITEM_ID"] = self.data["ITEM_ID"].astype(str).astype('float32')

        required_cols = ["OBJECT_ID", "REGION_ID", "REGION_ALIAS", "ITEM_ID", "PRODUCTLINE",
                         "TIMEFRAME_ID", "RESULT", "MP_ENABLED_ISO"]


        result_df = pd.merge(result_df, self.data, how="inner", on=["ITEM_ID", "time_idx"])

        return result_df

    def make_tft_predictions_df(self, best_tft, val_dataloader, new_pred, new_x, new_index):

        list1 = []
        maelist = []
        actual_data = [(y[0]) for x, y in iter(val_dataloader)]
        for i in list(new_index.index):
            pred_ts = new_pred[torch.tensor(i)].tolist()
            try:
                actuals = actual_data[0][i].tolist()
            except IndexError:
                pass
            # print(len(pred_ts))
            decoder_length1 = new_x["decoder_time_idx"][i].tolist()
            # print(len(decoder_length1))
            ITEM_ID = []
            ITEM_ID.append(new_index["ITEM_ID"][i])
            ITEM_ID = ITEM_ID * self.max_prediction_length
            # *new_x["decoder_time_idx"].size()[1] # multipying by  26
            # print(len(ITEM_ID))
            zipped = list(zip(actuals, pred_ts, decoder_length1, ITEM_ID))
            subtracted = [abs(x1 - x2) for (x1, x2) in zip(actuals, pred_ts)]
            mae = np.array(subtracted).mean()
            maelist.append(mae)
            print(f"{ITEM_ID[0]}  = {mae}")
            # print(list(zipped))

            [list1.append(item) for item in zipped]

            df_tft = pd.DataFrame(list1, columns=["ACTUALS", 'RESULT', 'time_idx', "ITEM_ID"])
            df_tft["ITEM_ID"] = df_tft["ITEM_ID"].astype(str).astype("float32")
            # RAW_COLUMN_LIST = ["PAI_REGION_ID","REGION_ALIAS","ITEM_ID","OBJECT_ID","PAI_TIMEFRAME_ID","RESULT","MP_ENABLED","PRODUCTLINE"]
        return df_tft





        # todo: pickling var_collection to reload again
        # todo: lagged visibility
        # todo: reference date is important for back testing
