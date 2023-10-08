import pandas as pd
import numpy as np
import math

class ImputeMissingValues:

    def __init__(self, config):
        self.config = config
        #sensor readout columns
        self.sensor_columns = ['GeffRef', 'GeffTest', 'IscRef', 'IscTest',
                          'TempRef', 'TempTest']


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
        daywise_nan_df = df.groupby(df['TIMESTAMP'].dt.date).apply(lambda x: x.isna().sum())
        nan_df = daywise_nan_df.drop("TIMESTAMP", axis=1)
        nan_df['TIMESTAMP'] = nan_df.index
        nan_df['TIMESTAMP'] = pd.to_datetime(nan_df['TIMESTAMP'])
        nan_df.index = pd.to_datetime(nan_df.index)
        return nan_df

    def impute_missing_values(self, site_df):
        site_df.reset_index(inplace=True)
        #daywise_nan_df gives counts of all the nans daywise
        daywise_nan_df = self.get_nan_counts(site_df)
        # Excluding rows where all columns are zero
        daywise_nan_df = daywise_nan_df[(daywise_nan_df != 0).any(axis=1)]

        #todo:Veify is this the right place for removing these Iscref rows
        # apply IscRef threshold
        site_df["IscRef"] = site_df["IscRef"].astype(float)
        site_df = site_df[site_df["IscRef"] > 0.6]

        #Intraday
        site_df = self.fill_intraday(site_df, daywise_nan_df)



        #Interday
        aggregated_site_df = self.resample_and_aggregate(site_df)
        aggregated_site_df = self.fill_interday(aggregated_site_df, daywise_nan_df)


        # intraday_lst = intraday_info_df.index.tolist()
        # interday_lst = set(daywise_nan_df.index.tolist())-set(intraday_lst)

        # Step 2: Split days into two sets based on the threshold of 360 nan counts
        less_than_threshold = []
        greater_than_threshold = []
        #todo get threshold in config

        intraday_nan_thrshld = 360
        return aggregated_site_df

    def fill_interday(self, site_df, daywise_nan_df):
        site_df["TIMESTAMP"] = site_df.index
        site_df['Month'] = site_df['TIMESTAMP'].dt.to_period('M')
        month_df = site_df.groupby('Month')[self.sensor_columns].mean().reset_index()
        month_df.ffill(inplace=True)

        ic(month_df.iloc[:,:4])
        ic(month_df.iloc[:, 4:])

        site_df = pd.merge(site_df, month_df, how="left", on=["Month"], suffixes=("_orig", "_filled"))

        for col in self.sensor_columns:
            col_orig = col + "_orig"
            col_filled = col + "_filled"
            # todo: Why is the null value count increasing after replacing the original values
            print(site_df[col_orig].isnull().sum())

            site_df[col_orig] = np.where(site_df[col_filled].notnull(), site_df[col_filled], site_df[col_orig])
            print(site_df[col_orig].isnull().sum())

            site_df.drop(columns=[col_filled], inplace=True)
            site_df.rename(columns={col_orig: col}, inplace=True)






        return site_df



    def resample_and_aggregate(self, df):
        """
        Resamples a DataFrame at the specified frequency and aggregates columns using specific functions.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            frequency (str): The frequency at which to resample the DataFrame (e.g., 'D' for daily).

        Returns:
            pandas.DataFrame: The resampled and aggregated DataFrame.
        """
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df.set_index("TIMESTAMP", inplace=True)
        # df.index = pd.to_datetime(df.index)
        # Resample the dataframe at the specified frequency
        resampled_df = df.resample("D")
        df[self.sensor_columns] = df[self.sensor_columns].astype(float)

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
        final_df = df.resample("D")[self.sensor_columns].agg(agg_functions)

        return final_df





    def fill_intraday(self, site_df, daywise_nan_df):
        """
        1. If < limit of 3 hr (360) --> ffill & bfill
        2. If > 3hr (360) and < 1 day (1440) --> average of previous and next day at that time
            1 day to 3 days  average of previous day and next day
        4 days to any days --> resample the data to hourly basis, fill with average of the whole month



        Args:
            site_df:
            daywise_nan_df:

        Returns:

        """

        intraday_nan_thrshld = self.config.get_custom_param("imputemissingvalues")['intraday_nan_count_thrshld']
        intra_ffill_limit = self.config.get_custom_param("imputemissingvalues")['intra_ffill_limit']



        #todo:Remove TIMESTAMP and other un important columns that


        for col in self.sensor_columns:

            # If < limit of 3 hr (360) --> ffill & bfill
            ffill_bool_df1 = daywise_nan_df.loc[(daywise_nan_df[col] > 0) & (daywise_nan_df[col] < 360), ["TIMESTAMP", col]]
            ffill_bool_df1['TIMESTAMP'] = pd.to_datetime(ffill_bool_df1['TIMESTAMP'])
            ffill_days_lst1 = ffill_bool_df1["TIMESTAMP"].dt.date.unique()


            intraday_data_df1 = site_df.loc[site_df["TIMESTAMP"].dt.date.isin(ffill_days_lst1), ['TIMESTAMP', col]]

            intraday_data_df1.set_index('TIMESTAMP',inplace=True)
            # intraday_data_df1["TIMESTAMP"] = intraday_data_df1.index
            # Forward-fill the first half of missing values
            filled_df1 = intraday_data_df1[col].groupby(pd.Grouper(freq='D')).ffill(limit= int(ffill_bool_df1[col].max()/2))
            # Backward-fill the second half of missing values
            filled_df1 = filled_df1.to_frame(name=col)
            filled_df1 = filled_df1[col].groupby(pd.Grouper(freq='D')).bfill(limit=int(ffill_bool_df1[col].max()/2))
            site_df = pd.merge(site_df, filled_df1, how="left", on=["TIMESTAMP"], suffixes=("_orig", "_filled"))

            col_orig = col + "_orig"
            col_filled = col + "_filled"
            site_df[col_orig] = np.where(site_df[col_filled].notnull(), site_df[col_filled], site_df[col_orig])


            site_df.drop(columns=[col_filled], inplace=True)
            site_df.rename(columns={col_orig: col}, inplace=True)



        return site_df







    def fill_sandwiching_days(self):
        pass
        # *************If > 3hr (360) and < 1 day (1440) --> average of previous and next day at that time

        # Create a new DataFrame with rows where the value is greater than 360 for any column

        # Calculate nan count for the previous day and next day
    #     daywise_nan_df['prev_day_nan'] = daywise_nan_df[col].shift(1)
    #     daywise_nan_df['next_day_nan'] = daywise_nan_df[col].shift(-1)
    #
    #     # Filter days based on conditions
    #     filtered_days = daywise_nan_df[(daywise_nan_df[col] >= intra_ffill_limit) &
    #                                    (daywise_nan_df['prev_day_nan'] == 0) &
    #                                    (daywise_nan_df['next_day_nan'] == 0)]['TIMESTAMP'].dt.date.unique()
    #
    #     # Create a new DataFrame with only the selected days
    #     filtered_daywise_nan_df = daywise_nan_df[daywise_nan_df['TIMESTAMP'].dt.date.isin(filtered_days)].copy()
    #
    #     # Drop the temporary columns
    #     filtered_daywise_nan_df.drop(['prev_day_nan', 'next_day_nan'], axis=1, inplace=True)
    #
    #     ffill_days_lst2 = filtered_daywise_nan_df["TIMESTAMP"].dt.day.unique()
    #     intraday_data_df2 = site_df[site_df["TIMESTAMP"].dt.day.isin(ffill_days_lst2)]
    #
    #     # Fill with the average of previous and next day at that time
    #
    #     for day in filtered_daywise_nan_df['TIMESTAMP'].dt.date.unique():
    #         day_data = site_df[site_df['TIMESTAMP'].dt.date == day][col]
    #
    #         # Get the value from the previous day at the same timestamp
    #         site_df['TIMESTAMP'] = pd.to_datetime(site_df['TIMESTAMP'])
    #         previous_day_value = site_df[site_df['TIMESTAMP'] == day_data.index[0] - pd.Timedelta(days=1)][col].iloc[0]
    #         next_day_value = site_df[site_df['TIMESTAMP'] == day_data.index[0] + pd.Timedelta(days=1)][col].iloc[0]
    #         #
    #         #     # Fill NaN values with the mean of the previous day and next day values
    #         site_df.loc[site_df['TIMESTAMP'].dt.date == day_data, col].fillna((previous_day_value + next_day_value) / 2,
    #                                                                           inplace=True)
    #     #
    #     #
    #     #
    #     #
    #     #
    #
    #     # todo consider the mean, medium and mode for filling -- if they can give better result.
    #
    # # intraday_info_df.loc[]
    # # todo replace for loop by groupby
    # # for day in intraday_lst:
    # #     daily_nan_counts = nan_counts_result.loc[day]
    # #     if all(count < nan_count_threshold for count in daily_nan_counts):
    # #         less_than_threshold.append(day)
    # #     else:
    # #         greater_than_threshold.append(day)
