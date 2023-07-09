from datetime import datetime
import numpy as np
import pandas as pd
class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def add_features(self):
        df['date'] = df['TIMESTAMP'].dt.date
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    # todo: Add date, days bu gk

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

    #todo: Group by productline, shopass,  seasonality
    # todo: the way averages over item_id and Productline are wrong. There is a leakage here
    #todo: Recession should be included
    #todo: Adding weights column and using it for covariate shift

    def quantile_normalize(self, df):
        """
        input: dataframe with numerical columns
        output: dataframe with quantile normalized values
        """
        df_sorted = pd.DataFrame(np.sort(df.values,
                                         axis=0),
                                 index=df.index,
                                 columns=df.columns)
        df_mean = df_sorted.mean(axis=1)
        df_mean.index = np.arange(1, len(df_mean) + 1)
        df_qn = df.rank(method="min").stack().astype(int).map(df_mean).unstack()
        return (df_qn)

    def add_features(self):
        """Add additional features"""
        #todo: add holidays and also covid data given by Adrian

        self.data["time_idx"] = (round(((self.data["YEAR"] - self.data["YEAR"].min())))).astype(int)
        self.data["METRIC"] = (self.data["YES"] + self.data["NO"]) * (self.data["YES"] - self.data["NO"])

        pivoted_data = self.data.pivot(index="YEAR", columns="SYSTEM", values="METRIC")


        quant_norm_pivot_data = self.quantile_normalize(pivoted_data)

        quant_norm_pivot_data["YEAR"] = quant_norm_pivot_data.index

        quant_norm_data = pd.melt(quant_norm_pivot_data, id_vars=['YEAR'], var_name=['SYSTEM'])
        quant_norm_data.rename({"value": "QUANTILE_NORM"}, axis=1, inplace=True)

        self.data  = pd.merge(quant_norm_data, self.data, how='inner', on=["YEAR", "SYSTEM"])



        #self.data["log_QUANTILE_NORM"] = np.log1p(self.data.QUANTILE_NORM)
        self.data["avg_QUANTILE_NORM_by_SYSTEM"] = self.data.groupby(["time_idx", "SYSTEM"],
                                                  observed=True).QUANTILE_NORM.transform("mean")


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


    def lagged_features(self):
        #todo: check this lagged code, there can be a bug
        #todo: Adding any other useful lag
        #lag1 feature
        self.data = self.data.set_index(["SYSTEM"]).sort_values("time_idx")
        self.data["QUANTILE_NORM_lagged_1"] = self.data["QUANTILE_NORM"].shift(periods=1)

        self.data["QUANTILE_NORM_lagged_1"].fillna(method='ffill', inplace=True)
        self.data["QUANTILE_NORM_lagged_1"].fillna(method='bfill', inplace=True)

        # lag2 feature
        self.data["QUANTILE_NORM_lagged_2"] = self.data["QUANTILE_NORM"].shift(periods=2)
        self.data["QUANTILE_NORM_lagged_2"].fillna(method='ffill', inplace=True)
        self.data["QUANTILE_NORM_lagged_2"].fillna(method='bfill', inplace=True)
        self.data.reset_index(inplace=True)

    def feature_engineer(self):
        self.add_features()
        self.lagged_features()
        self.check_inf_or_nan()
        return self.data


#todo: Feature engineered values are given as future unknown, however, if given as future known then the accuracy will improve.