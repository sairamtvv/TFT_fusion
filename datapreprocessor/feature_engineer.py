from datetime import datetime
import numpy as np
import pandas as pd
class FeatureEngineering:
    def __init__(self, data):
        self.data = data


    # todo: Add temperature, holidays, sensex,  covid data from adrian,style id
    # todo: why are product line averages not added and grouped prediction shall be better

    #todo: Group by productline, shopass,  seasonality
    # todo: the way averages over item_id and Productline are wrong. There is a leakage here
    #todo: Recession should be included
    #todo: Adding weights column and using it for covariate shift

    def assigning_weights(self):
        """Back up the original columns incase we require later"""
        # self.data["DEMANDPOT_ORIGINAL"] = self.data.DEMAND_POT
        # self.data["VISIBILITY_AVG_PCT_ORIGINAL"] = self.data.VISIBILITY_AVG_PCT
        self.data["weight"] = self.data.VISIBILITY_AVG_PCT
        self.data.loc[self.data["weight"] >= 0.5, "weight"] = 1.0



    def add_features(self):
        """Add additional features"""
        #todo: add holidays and also covid data given by Adrian
        self.data['covid'] = self.data['FIRST_DAY_OF_TIMEFRAME'].apply(
            lambda x: 1 if x > datetime(2020, 1, 20) and x < datetime(2022, 1, 1) else 0)
        self.data["time_idx"] = (round(((self.data["FIRST_DAY_OF_TIMEFRAME"] - self.data[
            "FIRST_DAY_OF_TIMEFRAME"].min()).dt.days) / 7)).astype(int)

        self.data["log_DEMAND_POT"] = np.log1p(self.data.DEMAND_POT)
        self.data["log_DEMAND_POT"].clip(lower=0, inplace=True)
        self.data["avg_DEMAND_POT_by_item"] = self.data.groupby(["time_idx", "ITEM_ID"],
                                                  observed=True).DEMAND_POT.transform("mean")
        self.data["avg_DEMAND_POT_by_productline"] = self.data.groupby(["time_idx", "PRODUCTLINE"],
                                               observed=True).DEMAND_POT.transform("mean")


    def check_inf_or_nan(self,  col_lst=("log_DEMAND_POT", "demand_pot_lagged_1","demand_pot_lagged_2" ) ):

        #todo: Add checking for inf also
        """"
        Checking final frame for nans and infinity
        These columns ("DEMAND_PCS", "VISIBILITY_AVG_PCT", "DEMAND_POT") definitely should not have NAN
        """
        for col_name in col_lst:
            nans_in_col = self.data[col_name].isnull().sum()
            ic(nans_in_col)
            assert (nans_in_col == 0), f"{col_name} should not have NAN values but {nans_in_col} found"


    def lagged_features(self):
        #todo: check this lagged code, there can be a bug
        #todo: Adding any other useful lag
        #lag1 feature
        self.data = self.data.set_index(["PRODUCTLINE", "ITEM_ID"]).sort_values("time_idx")
        self.data["demand_pot_lagged_1"] = self.data["DEMAND_POT"].shift(periods=1)

        self.data["demand_pot_lagged_1"].fillna(method='ffill', inplace=True)
        self.data["demand_pot_lagged_1"].fillna(method='bfill', inplace=True)

        # lag2 feature
        self.data["demand_pot_lagged_2"] = self.data["DEMAND_POT"].shift(periods=2)
        self.data["demand_pot_lagged_2"].fillna(method='ffill', inplace=True)
        self.data["demand_pot_lagged_2"].fillna(method='bfill', inplace=True)
        self.data.reset_index(inplace=True)

    def feature_engineer(self):
        self.assigning_weights()
        self.add_features()
        self.lagged_features()
        self.check_inf_or_nan()
        return self.data


#todo: Feature engineered values are given as future unknown, however, if given as future known then the accuracy will improve.