# -*- coding: utf-8 -*-
#"""DataFetcherLogger for Pytorch Forecasting TFT"""
import logging
import sys
import os


from icecream import install, ic
from pa_core.config import TrainTestPipelineConfig

from pa_demand_forecast.raw_forecast_tft.datapreprocessor.feature_engineer import \
    FeatureEngineering
from pa_demand_forecast.raw_forecast_tft.datapreprocessor.impute_missing import ImputeMissingValues

ic.configureOutput(prefix='TFT -> ')
install()
ic.configureOutput(includeContext=True, contextAbsPath=False)

#sys.path.append(str(Path(os.getcwd()).parent.parent))



# from impute_missing import ImputeMissingValues
# from feature_engineer import FeatureEngineer


class DataPreProcessor:

    def __init__(self, data, train_test_config):
        """
            Data Preprocessor takes care of three processes
            1) Making data subset
            2) Imputation
            3) Feature Engineering

        """
        self.data = data
        self.train_test_config = train_test_config

        # Parameters imported from YAML file
        self.col_type = None         # N or S
        self.has_minimum_timestamps = None
        self.Minimum_timestamps = None
        self.is_real_prediction = self.train_test_config.get_custom_param("is_real_prediction")
        for key, value in self.train_test_config.get_custom_param("datapreprocessor").items():
            setattr(self, key, value)



    def choose_N_or_S(self):
        pass

    def choose_data_with_min_timestamps(self):
        pass

    def choose_number_of_itemids(self):
        #todo: redo this code
        pass
        #
        # lst_ids = list(self.data.ITEM_ID.unique())
        # float_lst_ids = list(map(float, lst_ids))
        # float_lst_ids = float_lst_ids[:100]

        # new_data_frame_bool = self.data['ITEM_ID'].isin(float_lst_ids)
        # self.data = self.data[new_data_frame_bool]
        # ic(self.data.ITEM_ID.nunique())
        # # self.data = self.data.loc[data['ITEM_ID'].apply(lambda x: x in lst_ids)]


    def unforeseen_cut_over_data(self):
        pass

    def impute_missing(self):
        imputer_obj = ImputeMissingValues(self.data, self.train_test_config)
        self.data = imputer_obj.impute_missing_values()

    def real_prediction_fill_missing_vis(self):
        """
        Only when it is real prediction, when actual visibility is unknown,
        using the prediction length given in the config file,
        take the max time stamp and asssign the future visibility to 1

        """
        if self.is_real_prediction:
            training_cutoff = self.data["time_idx"].max() - self.max_prediction_length
            self.data.loc[self.data["time_idx"] > training_cutoff, "VISIBILITY_AVG_PCT"] = 1


    def feature_engineer(self):
        feat_eng_obj = FeatureEngineering(self.data)
        self.data = feat_eng_obj.feature_engineer()

    def choose_subset_data(self):
        self.choose_N_or_S()
        self.choose_data_with_min_timestamps()
        self.choose_number_of_itemids()
        self.unforeseen_cut_over_data()

    def save_filled_dataframe(self):
        "Persisit the dataframe as the mode filling is a time consuming task"
        # todo decide the file name
        self.data.to_pickle(self.save_imputed_featureengineered_df_filename, compression="zip")


    def preprocess_data(self):

       self.choose_subset_data()

       self.impute_missing()
       self.feature_engineer()
       self.real_prediction_fill_missing_vis()
       # if self.save_imputed_featureengineered_df:
       #     self.save_filled_dataframe()
       return self.data








