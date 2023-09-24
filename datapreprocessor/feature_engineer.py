from datetime import datetime
import numpy as np
import pandas as pd
class FeatureEngineering:
    def __init__(self, config, resampled_df):
        self.config = config
        self.resampled_df = resampled_df

    # todo: Adding weights column and using it for covariate shift
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

    def features_per_day(self):

        df_day = df.loc[df["time_idx"] == day]
        sum_day_IscRef = df_day["IscRef"].sum()
        sum_day_IscTest = df_day["IscTest"].sum()
        difference = sum_day_IscRef - sum_day_IscTest
        deviation = (sum_day_IscRef - sum_day_IscTest) * 100 / sum_day_IscRef
        baseline_loss = -0.946
        soiling_loss = deviation + 0.946
        lst_IscRef.append(sum_day_IscRef)
        lst_IscTest.append(sum_day_IscTest)
        lst_diff.append(difference)
        lst_percent_dev.append(deviation)
        lst_percent_soilingloss.append(soiling_loss)

        mean_day_GeffRef = df_day["GeffRef"].mean()
        mean_day_TempRef = df_day["TempRef"].mean()
        lst_GeffRef.append(mean_day_GeffRef)
        lst_TempRef.append(mean_day_TempRef)



    def check_inf_or_nan(self,  col_lst=None ):

        #todo: Add checking for inf also
        """"
        Checking final frame for nans and infinity
        These columns ("DEMAND_PCS", "VISIBILITY_AVG_PCT", "DEMAND_POT") definitely should not have NAN
        """
        if col_lst is None:
            col_lst = ("avg_QUANTILE_NORM_by_SYSTEM","QUANTILE_NORM_lagged_1","QUANTILE_NORM_lagged_2" )

        for col_name in col_lst:
            nans_in_col = self.data[col_name].isnull().sum()
            ic(nans_in_col)
            assert (nans_in_col == 0), f"{col_name} should not have NAN values but {nans_in_col} found"



    def feature_engineer(self):
        self.add_features()
        self.lagged_features()
        self.check_inf_or_nan()
        return self.data


#todo: Feature engineered values are given as future unknown, however, if given as future known then the accuracy will improve.