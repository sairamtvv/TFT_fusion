# -*- coding: utf-8 -*-
#"""DataFetcherLogger for Pytorch Forecasting TFT"""
import logging
import sys
import os
import pandas as pd

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

        # # Parameters imported from YAML file
        # self.col_type = None         # N or S
        # self.has_minimum_timestamps = None
        # self.Minimum_timestamps = None
        # self.is_real_prediction = self.train_test_config.get_custom_param("is_real_prediction")
        # for key, value in self.train_test_config.get_custom_param("datapreprocessor").items():
        #     setattr(self, key, value)

    def print_info_on_df(self, df, site=None, stage="initial"):

        # print(f"For {site} after {stage.capitalize()} rows are {len(df)}")

        missing_rows = df["IscRef"].isnull().sum()
        missing_percentage = (missing_rows / len(df)) * 100
        print(f"For {site} after ***{stage.capitalize()}*** missingrows ={missing_rows} "
              f"totalrows= {len(df)} percentage_miss={missing_percentage}")
        # print(f"For {site} after {stage.capitalize()} missing percentage is {missing_percentage} out of {total_rows}")
        # print the percentage of nulls also



    def connvert_to30s_freq(self):
        self.data['TIMESTAMP'] = pd.to_datetime(self.data['TIMESTAMP'])
        locations = self.data["location"].unique()[-1]
        locations_lst = []
        locations_lst.append(locations)

        self.thirty_sec_df = pd.DataFrame()
        for loc_no, site in enumerate(list(locations_lst)):
            site_df = self.data.loc[self.data["location"] == site].copy()
            total_rows = len(site_df)
            self.print_info_on_df(site_df, site, "initial site dataframe")
            site_df.set_index("TIMESTAMP", inplace=True)
            site_df = site_df.loc[~site_df.index.duplicated(keep='first')]
            site_df = site_df.asfreq('30S')
            self.print_info_on_df(site_df, site, "after setting 30S frequency")

            # filling missing data for night for column Iscref #436314 are night values
            site_df.loc[((site_df.index.hour > 18) | (site_df.index.hour < 6)) & (site_df["IscRef"].isna()), "IscRef"] =0


            self.print_info_on_df(site_df, "site" "after filling night data with zero")

            site_df["IscRef"] = site_df["IscRef"].fillna(method='ffill', limit=10)
            self.print_info_on_df(site_df, site, "after filling with ffill with limit=10")

            site_df["IscRef"] =site_df["IscRef"].fillna(method='bfill', limit=10)
            self.print_info_on_df(site_df, site, "after filling with bfill with limit=10")

            #todo:Understand and remove this
            #temporarily making all nans of IscRef to be zero
            site_df.loc[site_df["IscRef"].isna(),"IscRef"] = 0
            self.print_info_on_df(site_df, site, "temporarily made iscnan to zero")


            #apply IscRef threshold
            site_df["IscRef"] = site_df["IscRef"].astype(float)
            thresh_index = site_df[site_df["IscRef"]<0.6].index
            site_df.drop(index=thresh_index, inplace=True)
            self.print_info_on_df(site_df, site, "apply Iscref threshold 0.6")

            # todo:Understand and remove this  2nd time
            # temporarily making all nans of IscRef to be zero
            site_df.loc[site_df["IscRef"].isna(), "IscRef"] = 0
            self.print_info_on_df(site_df, site, "temporarily made iscnan to zero")


            #Needto fill the nans that are left behind in the other columns
            self.thirty_sec_df = pd.concat([self.thirty_sec_df, site_df])


    def preprocess_data(self):
        # Convert 'Timestamp' column to datetime
        self.connvert_to30s_freq()



        # Resample at 30-second intervals
        #df_resampled = df.resample('30S').mean()
        # #a=1
        # self.data = self.data.copy()

        # self.impute_missing()
        # self.feature_engineer()
        # self.choose_subset_data()
        # return self.data








