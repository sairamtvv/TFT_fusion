import pandas as pd
import mlflow

from configs import log
from configs.config import TrainTestPipelineConfig, LoggerConfig
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



class FusionForecast():

    def __init__(self, train_test_config):

        self.train_test_config = train_test_config
        self.logger = logging.getLogger(__name__)
        self.DATA_OUTPUT_DIR = Path(os.getcwd()) / "All_data" / self.train_test_config.get_custom_param("run_name")
        os.makedirs(self.DATA_OUTPUT_DIR, exist_ok=True)


        # Parameters imported from YAML file

        for key, value in self.train_test_config.get_custom_param("fusionforecast").items():
            setattr(self, key, value)

    def read_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data.reset_index(inplace=True)


    def impute_missing(self):
        pass

    def feature_engineer(self):
        feat_eng_obj = FeatureEngineering(self.data)
        self.data = feat_eng_obj.feature_engineer()

        mask = self.data["SCREENING_POTENTIAL"] > 10
        self.data = self.data.loc[mask]

        self.max_prediction_length = self.train_test_config.get_custom_param("structuringdataset")["max_prediction_length"]



        # shouldnt_see_data = self.data["time_idx"].max() - self.max_prediction_length
        # self.data = self.data.loc[self.data["time_idx"] < shouldnt_see_data]


        def get_datetime_string():
            return datetime.datetime.now().strftime("%m-%d-%Y__%H-%M-%S")

        # save train dataframe for debugging purposes:
        if self.train_test_config.get_custom_param("save_pickle_file"):
            file_path_name = f"{self.DATA_OUTPUT_DIR}/TFT_{get_datetime_string()}_train_df.pkl"
            self.data.to_pickle(file_path_name, compression="zip")
            mlflow.log_artifacts(f"{self.DATA_OUTPUT_DIR}")

    def train_model(self):
        tft_dfc_model_obj = TFTDFCModel(self.data, self.train_test_config)

        log_every_n_step = self.train_test_config.get_custom_param("log_every_n_step")
        log_models = self.train_test_config.get_custom_param("log_models")
        mlflow.pytorch.autolog(log_every_n_step=log_every_n_step, log_models=log_models)
        best_model_path = tft_dfc_model_obj.train()

    def predict(self):
        prefict_obj = Predict(self.data, self.train_test_config)





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
    # variable_collection = get_var_collection(
    #     config=pipeline_config, timeframe_name="train_1"
    # )

    #************Starts Here *********************************************************
        forecast_obj = FusionForecast(pipeline_config)
        forecast_obj.read_data()
        forecast_obj.impute_missing()
        forecast_obj.feature_engineer()

        forecast_obj.train_model()




