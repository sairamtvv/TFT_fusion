import pandas as pd
import mlflow

from configs import log
from configs.config import TrainTestPipelineConfig, LoggerConfig
from datapreprocessor.dataloader import DataLoader
from datapreprocessor.datapreprocess import DataPreProcessor
from datapreprocessor.feature_engineer import FeatureEngineering

import logging
import os
from pathlib import Path
import datetime

from icecream import install, ic

from model_builder_trainer.tft_dfc_model import TFTDFCModel
from predict.predict import Predict

ic.configureOutput(prefix='TFT -> ')
install()
ic.configureOutput(includeContext=True, contextAbsPath=False)



class SoilingLossForecast():


    def __init__(self, train_test_config):

        self.train_test_config = train_test_config
        self.logger = logging.getLogger(__name__)
        self.DATA_OUTPUT_DIR = Path(os.getcwd()) / "All_data" / self.train_test_config.get_custom_param("run_name")
        os.makedirs(self.DATA_OUTPUT_DIR, exist_ok=True)


        # Parameters imported from YAML file

        for key, value in self.train_test_config.get_custom_param("fusionforecast").items():
            setattr(self, key, value)

    def read_data(self):
        """
        This reads data from excel using dataloader and converrts to dataframe

        :return:
        """
        if self.train_test_config.get_custom_param("quick_debug"):
            # flder_pth = "Raw_data/AP/P-1_Karnal/Apr-23/"
            # data_loader = DataLoader(flder_pth, append_col=False)
            # self.data = data_loader.read_file("/home/sai/Pictures/TFT_fusion/Raw_data/AP/P-1_Karnal/Apr-23/P1_RawMeasurementData_2023-04-18T18-21.csv")

            self.data = pd.read_pickle("Raw_data/resampled_df_all_locations", compression="zip")


        else:
            flder_pth = "Raw_data/AP/"
            data_loader = DataLoader(flder_pth, append_col=True)
            self.data = data_loader.fetch_df(level_names_lst=["year_month", "location"])
        #
        # df = data_loader.fetch_df()
        # self.data = pd.read_csv(self.file_path)

        #logger.info("loaded the complete data")

        #Remove unnecessary rows and columns
        remove_cols = set(["Ref_Panel_washed", "Update_Offset", "year_month"]).intersection(set(self.data.columns))
        self.data.drop(columns=list(remove_cols), inplace=True)
        #todo:go deeper into what is getting  dropped
        #gurpreettodo:why are we dropping all the NA
        #self.data.dropna(how="any", inplace=True)

        self.data.reset_index(drop=True, inplace=True)
        # self.data.loc[self.data["TIMESTAMP"].isnull()].index



    def main(self, forecast_obj):
        forecast_obj.read_data()
        resampled_df = self.data.copy()

        if not self.train_test_config.get_custom_param("quick_debug"):
            preprocessor_obj = DataPreProcessor(self.data, self.train_test_config)
            preprocessor_obj.preprocess_data()
            resampled_df = preprocessor_obj.resampled_df
            resampled_df.to_pickle("resampled_df.pkl", compression="zip")


        tft_dfc_model_obj = TFTDFCModel(resampled_df, self.train_test_config)

        log_every_n_step = self.train_test_config.get_custom_param("log_every_n_step")
        log_models = self.train_test_config.get_custom_param("log_models")
        #mlflow.pytorch.autolog(log_every_n_step=log_every_n_step, log_models=log_models)
        best_model_path = tft_dfc_model_obj.train()

        return resampled_df





    # def feature_engineer(self):
    #     feat_eng_obj = FeatureEngineering(self.data)
    #     self.data = feat_eng_obj.feature_engineer()
    #
    #     mask = self.data["SCREENING_POTENTIAL"] > 10
    #     self.data = self.data.loc[mask]
    #
    #     self.max_prediction_length = self.train_test_config.get_custom_param("structuringdataset")["max_prediction_length"]
    #
    #
    #
    #     shouldnt_see_data = self.data["time_idx"].max() - self.max_prediction_length
    #     self.data = self.data.loc[self.data["time_idx"] < shouldnt_see_data]
    #
    #
    #     def get_datetime_string():
    #         return datetime.datetime.now().strftime("%m-%d-%Y__%H-%M-%S")
    #
    #     # save train dataframe for debugging purposes:
    #     if self.train_test_config.get_custom_param("save_pickle_file"):
    #         file_path_name = f"{self.DATA_OUTPUT_DIR}/TFT_{get_datetime_string()}_train_df.pkl"
    #         self.data.to_pickle(file_path_name, compression="zip")
    #         mlflow.log_artifacts(f"{self.DATA_OUTPUT_DIR}")
    #
    # def train_model(self):

    # def predict(self):
    #     prefict_obj = Predict(self.data, self.train_test_config)
    #
    #




if __name__ == "__main__":
    config_list = ["./pipeline_config_TFT.yaml"]
    logger_config_list = ["./pipeline_config_TFT.yaml"]

    pipeline_config = TrainTestPipelineConfig(config_list)
    logger_config = LoggerConfig(logger_config_list)

    log.setup_logging(logger_config)
    logger = logging.getLogger(__name__)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger.info("ML pipeline started")

    # *************************MLFLOW START ********************************
    # defining a new experiment
    experiment_name = pipeline_config.get_custom_param("mlflow_experiment_name")
    my_custom_tag = pipeline_config.get_custom_param("my_custom_tag")
    run_description = pipeline_config.get_custom_param("run_description")
    # returns experiment ID
    try:
        # creating a new experiment
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    with mlflow.start_run(nested=True, experiment_id=exp_id,
                          run_name=pipeline_config.get_custom_param("run_name"),
                          tags={
                              "My_custom_tag": my_custom_tag,
                              "mlflow.note.content": run_description}
                          ):

        forecast_obj = SoilingLossForecast(pipeline_config)
        forecast_obj.main(forecast_obj)


    #************Starts Here *********************************************************

        # forecast_obj.impute_missing()
        # forecast_obj.feature_engineer()
        #
        # forecast_obj.train_model()




