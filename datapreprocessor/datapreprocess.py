# -*- coding: utf-8 -*-
#"""DataFetcherLogger for Pytorch Forecasting TFT"""
import logging
import sys
import os
import pandas as pd

#sys.path.append(str(Path(os.getcwd()).parent.parent))
import numpy as np


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

    import pandas as pd

    def resample_and_aggregate(self, df):
        """
        Resamples a DataFrame at the specified frequency and aggregates columns using specific functions.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            frequency (str): The frequency at which to resample the DataFrame (e.g., 'D' for daily).

        Returns:
            pandas.DataFrame: The resampled and aggregated DataFrame.
        """

        # Resample the dataframe at the specified frequency
        resampled_df = df.resample("D")

        # Define the aggregation functions for each column
        agg_functions = {
            'GeffRef': 'mean',
            'GeffTest': 'mean',
            'IscRef': 'sum',
            'IscTest': 'sum',
            'TempRef': 'mean',
            'TempTest': 'mean',
            #'location': 'first' # Use 'first' or 'last' to retain the constant value
             }

        # Apply the aggregation functions to the resampled dataframe
        final_df = resampled_df.agg(agg_functions)

        return final_df


    def typecast_columns(self):

        self.numerical_lst = ['GeffRef', 'GeffTest', 'IscRef', 'IscTest', 'TempRef', 'TempTest']
        def make_float(num):
            try:
                num = float(num)
            except:
                num = np.nan
            return num

        for num_col in self.numerical_lst:
            self.data[num_col] = self.data[num_col].apply(make_float)


        # self.catergorical_lst = ["location"]
        #
        # def make_str(cat):
        #     try:
        #         cat = float(cat)
        #     except:
        #         cat = np.nan
        #     return cat
        #
        # for cat_col in self.catergorical_lst:
        #     self.data[cat_col] = self.data[cat_col].apply(make_str)


    def print_nans_per_day(self, df):
        """
        #Assuminf there is a column of date

        Args:
            df:

        Returns:

        """
        df = df.copy()
        df.set_index("date").unstack().isnull().sum()




    def prepare_data(self):
        self.typecast_columns()
        self.data['TIMESTAMP'] = pd.to_datetime(self.data['TIMESTAMP'])
        locations_lst = self.data["location"].unique()
        # locations_lst = []
        # locations_lst.append(locations)

        self.resampled_df = pd.DataFrame()
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


            #We have to order the index in the ascending order before ffill or bfill

            #site_df.index = site_df.sort_index()
            site_df = site_df.fillna(method='ffill', limit=10)
            self.print_info_on_df(site_df, site, "after filling with ffill with limit=10")

            site_df =site_df.fillna(method='bfill', limit=10)
            self.print_info_on_df(site_df, site, "after filling with bfill with limit=10")

            #todo:Understand and remove this
            #temporarily making all nans of IscRef to be zero
            site_df.loc[site_df["IscRef"].isna(),"IscRef"] = 0
            self.print_info_on_df(site_df, site, "temporarily made iscnan to zero")


            #apply IscRef threshold
            site_df["IscRef"] = site_df["IscRef"].astype(float)
            site_df = site_df[site_df["IscRef"]>0.6]
            # thresh_index = site_df[site_df["IscRef"]<0.6].index
            # site_df.drop(index=thresh_index, inplace=True)
            self.print_info_on_df(site_df, site, "apply Iscref threshold 0.6")

            # todo:Understand and remove this  2nd time
            # temporarily making all nans of IscRef to be zero
            site_df.loc[site_df["IscRef"].isna(), "IscRef"] = 0
            self.print_info_on_df(site_df, site, "temporarily made iscnan to zero")


            site_df[self.numerical_lst] = site_df[self.numerical_lst].fillna(0)

            site_df[self.numerical_lst] = site_df[self.numerical_lst].astype(float)

            

            resampled_day_loc_df = self.resample_and_aggregate(site_df)
            resampled_day_loc_df["location"] = site

            #making numbers as the index instead og TIMESTAMP
            resampled_day_loc_df.reset_index(inplace=True)

            #Needto fill the nans that are left behind in the other columns
            self.resampled_df = pd.concat([self.resampled_df, resampled_day_loc_df])

        self.resampled_df = self.resampled_df[~self.resampled_df.isna()]


    def asssign_soilingloss(self):
        self.resampled_df["difference"] =  self.resampled_df["IscRef"] -  self.resampled_df["IscTest"]
        self.resampled_df["deviation"] = (self.resampled_df["difference"]  * 100) / self.resampled_df["IscRef"]
        baseline_loss = -0.946
        self.resampled_df["soiling_loss"] = self.resampled_df["deviation"] - baseline_loss









    def add_features(self):
        """Add additional features"""

        # Extract the date from datetime
        self.resampled_df['date'] = self.resampled_df['TIMESTAMP'].dt.date
        self.resampled_df["time_idx"] = (self.resampled_df["date"] - self.resampled_df["da"
                                                                                       "te"].min()).apply(lambda x: x.days).astype(int)

        #self.resampled_df["log_soiling_loss"] = np.log1p(self.elf.resampled_df["soiling_loss"])

        # todo: This average taken below can have data leakage
        self.resampled_df["avg_soiling_loss_by_location"] = self.resampled_df.groupby(["time_idx", "location"],
                                                                observed=True)["soiling_loss"].transform("mean")


        self.resampled_df["REFERENCE_MONTH"] = self.resampled_df['TIMESTAMP'].dt.month
        self.resampled_df["REFERENCE_WEEK"]  = self.resampled_df['TIMESTAMP'].dt.isocalendar().week

    def lagged_features(self):
        # todo: check this lagged code, there can be a bug
        # todo: Adding any other useful lag
        # lag1 feature
        self.resampled_df = self.resampled_df.set_index(["location"]).sort_values("time_idx")
        self.resampled_df["soiling_loss_lag_1"] = self.resampled_df["soiling_loss"].shift(periods=1)

        self.resampled_df["soiling_loss_lag_1"].fillna(method='ffill', inplace=True)
        self.resampled_df["soiling_loss_lag_1"].fillna(method='bfill', inplace=True)

        # lag2 feature
        self.resampled_df["soiling_loss_lag_2"] = self.resampled_df["soiling_loss"].shift(periods=2)
        self.resampled_df["soiling_loss_lag_2"].fillna(method='ffill', inplace=True)
        self.resampled_df["soiling_loss_lag_2"].fillna(method='bfill', inplace=True)
        self.resampled_df.reset_index(inplace=True)





    def preprocess_data(self):
        self.prepare_data()
        self.asssign_soilingloss()
        self.add_features()
        self.lagged_features()
        a = 1



        # Resample at 30-second intervals
        #df_resampled = df.resample('30S').mean()
        # #a=1
        # self.data = self.data.copy()

        # self.impute_missing()
        # self.feature_engineer()
        # self.choose_subset_data()
        # return self.data








