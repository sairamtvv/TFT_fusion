import pandas as pd
import mlflow
from datapreprocessor.feature_engineer import FeatureEngineering
from model_builder_trainer.tft_dfc_model import TFTDFCModel


class FusionForecast():

    def __init__(self, train_test_config):

        self.train_test_config = train_test_config

        # Parameters imported from YAML file

        for key, value in self.train_test_config.get_custom_param("imputemissingvalues").items():
            setattr(self, key, value)

    def read_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data.reset_index(inplace=True)

    def impute_missing(self):
        pass

    def feature_engineer(self):
        self.data = FeatureEngineering(self.data)

        # save train dataframe for debugging purposes:
        if train_test_config.get_custom_param("save_pickle_file"):
            file_path_name = f"{self.DATA_OUTPUT_DIR}/{EXPERIMENT_NAME}_{get_datetime_string()}_train_df.pkl"
            train_df.to_pickle(file_path_name, compression="zip")
            mlflow.log_artifacts(f"{self.DATA_OUTPUT_DIR}")

    def train_model(self):
        tft_dfc_model_obj = TFTDFCModel(self.data, self.train_test_config)

        log_every_n_step = train_test_config.get_custom_param("log_every_n_step")
        log_models = train_test_config.get_custom_param("log_models")
        mlflow.pytorch.autolog(log_every_n_step=log_every_n_step, log_models=log_models)
        best_model_path = tft_dfc_model_obj.train()






if __name__ == "__main__":

   forecast_obj = FusionForecast(train_test_config)
   forecast_obj.read_data()
   forecast_obj.impute_missing()
   forecast_obj.feature_engineer()
   forecast_obj.train_model()



