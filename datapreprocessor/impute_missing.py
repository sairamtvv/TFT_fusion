import pandas as pd
import numpy as np


class ImputeMissingValues:

    def __init__(self, config):
        self.config = config


        # Parameters imported from YAML file

        #for key, value in self.config.get_custom_param("imputemissingvalues").items():
        #    setattr(self, key, value)

    def fill_NAN_and_empty_string_with_npnan(self, df):
        # Filled the 'NAN' str and empty str with nan
        df[df == 'NAN'] = np.nan
        df[df == ''] = np.nan
        return df

    def fill_zeros_during_nighttime(self, site_df):
        # filling missing data for night for
        site_df = site_df[~((site_df.index.hour > 18) | (site_df.index.hour < 6))]
        return site_df

    def find_days_with_max_data(self, df):
        pass


    def fill_days_with_max_data(self):
        pass

    def get_nan_counts(self, site_df):
        site_df.reset_index(inplace=True)
        #todo: why is 1560 max counts instead of 1440 after removing night
        return site_df.groupby(site_df['TIMESTAMP'].dt.date).apply(lambda x: x.isna().sum())


    def impute_missing_values(self, site_df):
        # Step 1: Make a list of days for intraday
        daywise_nan_df = self.get_nan_counts(site_df)
        intraday_info_df = daywise_nan_df[daywise_nan_df.min(axis=1) < 2880]

        intraday_lst = intraday_info_df.index.tolist()
        interday_lst = set(daywise_nan_df.index.tolist())-set(intraday_lst)

        # Step 2: Split days into two sets based on the threshold of 360 nan counts
        less_than_threshold = []
        greater_than_threshold = []
        #todo get threshold in config

        intraday_nan_thrshld = 360
        return site_df
    def fill_intraday(self, df, intraday_info_df):
        intraday_nan_thrshld = self.config.get_custom_param("imputemissingvalues")['intraday_nan_count_thrshld']

        # intraday_info_df.loc[]
        # # todo replace for loop by groupby
        # for day in intraday_lst:
        #     daily_nan_counts = nan_counts_result.loc[day]
        #     if all(count < nan_count_threshold for count in daily_nan_counts):
        #         less_than_threshold.append(day)
        #     else:
        #         greater_than_threshold.append(day)
        return df

    def fill_interday(self, df, interday_lst):
       # # Step 3: Fill missing values for A) with ffill
       #  for day in less_than_threshold:
       #      site_df.loc[site_df['TIMESTAMP'].dt.date == day].fillna(method='ffill', inplace=True)
       #      #site_df.loc[site_df['TIMESTAMP'].dt.date == day] = site_df.loc[
       #      #    site_df['TIMESTAMP'].dt.date == day
       #      #    ].fillna(method='bfill')
       #
       #  # Step 4: Resample and ffill for B)
       #  for day in greater_than_threshold:
       #      daily_data = site_df[site_df['TIMESTAMP'].dt.date == day]
       #      daily_data.set_index('TIMESTAMP', inplace=True)
       #      resampled_data = daily_data.resample('1H').ffill()
       #      site_df.loc[site_df['TIMESTAMP'].dt.date == day] = resampled_data.reset_index()

        return df

    # def impute_missing_values(self):
    #     """ All the above functions shall be executed here"""
    #     if self.load_imputed_df:
    #         self.data = pd.read_pickle(self.load_imputed_df_name, compression="zip")
    #     else:
    #         self.choose_subset_data()
    #         #self.backup_original_columns()
    #         self.fill_missing_vis_by_mode() #todo: donot impute using mode
    #         self.fill_when_both_vis_pcs_nan(lower=self.lowest_visibility)
    #         self.fill_nan_pcs()
    #         self.clip_vis_and_pot(lower=self.lowest_visibility)
    #         self.origin_unknown_nans()
    #         self.assign_demandpot()
    #         self.check_inf_or_nan()
    #
    #         self.get_nan_counts()
    #         return self.data
