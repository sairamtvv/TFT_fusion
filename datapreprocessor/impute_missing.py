import pandas as pd
import numpy as np
import math

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
        site_df = site_df[~((site_df.index.hour > 17) | (site_df.index.hour < 6))]
        return site_df

    def find_days_with_max_data(self, df):
        pass


    def fill_days_with_max_data(self):
        pass

    def get_nan_counts(self, site_df):
        df = site_df.copy()
        #todo: why is 1560 max counts instead of 1440 after removing night
        return df.groupby(df['TIMESTAMP'].dt.date).apply(lambda x: x.isna().sum())

    def impute_missing_values(self, site_df):
        site_df.reset_index(inplace=True)
        #daywise_nan_df gives counts of all the nans daywise
        daywise_nan_df = self.get_nan_counts(site_df)
        # Excluding rows where all columns are zero
        daywise_nan_df = daywise_nan_df[(daywise_nan_df != 0).any(axis=1)]
        self.fill_intraday(site_df, daywise_nan_df)

        # intraday_lst = intraday_info_df.index.tolist()
        # interday_lst = set(daywise_nan_df.index.tolist())-set(intraday_lst)

        # Step 2: Split days into two sets based on the threshold of 360 nan counts
        less_than_threshold = []
        greater_than_threshold = []
        #todo get threshold in config

        intraday_nan_thrshld = 360
        return site_df
    def fill_intraday(self, site_df, daywise_nan_df):

        intraday_nan_thrshld = self.config.get_custom_param("imputemissingvalues")['intraday_nan_count_thrshld']
        intra_ffill_limit = self.config.get_custom_param("imputemissingvalues")['intra_ffill_limit']

        # 1. If < limit of 3 hr (360) --> ffill & bfill
        # 2. If > 3hr (360) and < 1 day (1440) --> average of previous and next day at that time
        #    1 day to 3 days  average of previous day and next day
        # 3. 4 days to any days --> resample the data to hourly basis, fill with average of the whole month

        for col in site_df.columns:

            # *************If < limit of 3 hr (360) --> ffill & bfill
            #if daywise_nan_df[col] > 0:
                # Calculate the limit as half of the missing count
             #   limit_mid = math.ceil(daywise_nan_df[col] / 2)
            ffill_bool_df1 = daywise_nan_df[(daywise_nan_df.iloc[:, :-1] < 360).any(axis=1)]
            ffill_days_lst1 = ffill_bool_df1["TIMESTAMP"].dt.day.unique()
            # Create a new DataFrame with rows where the value is greater than 360 for any column
            filtered_days_df = daywise_nan_df[(daywise_nan_df.iloc[:, :-1] > 360).any(axis=1)]

            intraday_data_df1 = site_df[site_df["TIMESTAMP"].dt.day.isin(ffill_days_lst1)]

            # Forward-fill the first half of missing values
            filled_df1 = intraday_data_df1.groupby(pd.Grouper(freq='D')).ffill(limit=limit_mid)
            # Backward-fill the second half of missing values
            filled_df1 = filled_df1.groupby(pd.Grouper(freq='D')).bfill(limit=limit_mid)

            # Update the original DataFrame with the filled values
            site_df.update(filled_df)

            # *************If > 3hr (360) and < 1 day (1440) --> average of previous and next day at that time

            # Calculate nan count for the previous day and next day
            daywise_nan_df['prev_day_nan'] = daywise_nan_df.groupby('column')[col].shift(1)
            daywise_nan_df['next_day_nan'] = daywise_nan_df.groupby('column')[col].shift(-1)

            # Filter days based on conditions
            filtered_days = daywise_nan_df[(daywise_nan_df[col] > intra_ffill_limit) &
                                           (daywise_nan_df['prev_day_nan'] == 0) &
                                           (daywise_nan_df['next_day_nan'] == 0)]['TIMESTAMP'].dt.date.unique()

            # Create a new DataFrame with only the selected days
            filtered_daywise_nan_df = daywise_nan_df[daywise_nan_df['TIMESTAMP'].dt.date.isin(filtered_days)].copy()

            # Drop the temporary columns
            filtered_daywise_nan_df.drop(['prev_day_nan', 'next_day_nan'], axis=1, inplace=True)

            #ffill_days_lst2 = filtered_daywise_nan_df["TIMESTAMP"].dt.day.unique()
            #intraday_data_df2 = site_df[site_df["TIMESTAMP"].dt.day.isin(ffill_days_lst2)]

            # Fill with the average of previous and next day at that time

            for day in filtered_daywise_nan_df['TIMESTAMP'].dt.date.unique():
                day_data = site_df[site_df['TIMESTAMP'].dt.date == day][col]

                # Get the value from the previous day at the same timestamp
                previous_day_value = site_df[site_df['TIMESTAMP'] == day_data.index[0] - pd.Timedelta(days=1)][col].iloc[0]
                next_day_value = site_df[site_df['TIMESTAMP'] == day_data.index[0] + pd.Timedelta(days=1)][col].iloc[0]

                # Fill NaN values with the mean of the previous day and next day values
                site_df.loc[site_df['TIMESTAMP'].dt.date == day_data, col].fillna((previous_day_value + next_day_value) / 2,
                                                                         inplace=True)






            # todo consider the mean, medium and mode for filling -- if they can give better result.

        # intraday_info_df.loc[]
        # todo replace for loop by groupby
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
