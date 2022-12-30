import pandas as pd
from pa_core.config import TrainTestPipelineConfig


class ImputeMissingValues:

    def __init__(self, data, train_test_config):
        self.data = data
        self.train_test_config = train_test_config


        # Parameters imported from YAML file

        for key, value in self.train_test_config.get_custom_param("imputemissingvalues").items():
            setattr(self, key, value)


    @property
    def get_dataframe(self):
        return self.data

    def choose_subset_data(self):
        pass




    # todo: Make the future Visibilities =1, probably now it is 0.5 now
    # todo: Make a new column of visibility and do all the changing there, small values of visibilities impacts

    def fill_missing_vis_by_mode(self):

        """
        Filling the VISIBILITY with the mode for an ITEM_ID
        """

        # TODO: Rich progress bar showing  no. of items left for mode filling

        ic("Filling mode has been observed to take time")
        ic(self.data.ITEM_ID.nunique())
        item_ids_nan_vis = list(
            set((self.data.loc[(self.data["VISIBILITY_AVG_PCT"].isnull()), "ITEM_ID"])))
        ic(len(item_ids_nan_vis))

        lst_no_mode = []
        for item_id in item_ids_nan_vis:
            mode_df = self.data.loc[(self.data["ITEM_ID"] == item_id)]["VISIBILITY_AVG_PCT"].agg(
                lambda x: x.value_counts())  # .index[0])
            if mode_df.empty:
                lst_no_mode.append(item_id)
            else:
                mode = mode_df.index[0]
                # print(f"{item_id}= {mode}")
                self.data.loc[(self.data["ITEM_ID"] == item_id) & (
                    self.data.VISIBILITY_AVG_PCT.isnull()), "VISIBILITY_AVG_PCT"] = mode

    def fill_when_both_vis_pcs_nan(self, lower=0.5):
        """Solving Problem of NANS in both DEMAND_PCS and VISIBILITY,
           Assigning DEMAND_PCS =0 and VISIBILITY=0.5"""
        self.data.loc[
            (self.data["DEMAND_PCS"].isnull()) & (self.data["VISIBILITY_AVG_PCT"].isnull()),
            ["DEMAND_PCS", "VISIBILITY_AVG_PCT"]] = [0, lower]

    def fill_nan_pcs(self):
        "Nans of demand_pcs are filled with zero"
        self.data.loc[(self.data["DEMAND_PCS"].isnull()), ["DEMAND_PCS"]] = 0

    def clip_vis_and_pot(self, lower=0.5):
        """Solving Problem of Infinity, whenever VIS=0  or small make it lower"""
        self.data["VISIBILITY_AVG_PCT"] = self.data["VISIBILITY_AVG_PCT"].clip(lower=lower)
        self.data["DEMAND_POT"].clip(lower=0, inplace=True)

    def origin_unknown_nans(self):
        # todo:Figure out why there are still NANS left
        item_ids_nan_vis = self.data.loc[(self.data["VISIBILITY_AVG_PCT"].isnull()), "ITEM_ID"]
        item_ids_nan_vis = list(item_ids_nan_vis)

        lst_no_mode = []
        for item_id in item_ids_nan_vis:
            mode_df = self.data.loc[(self.data["ITEM_ID"] == item_id)]["VISIBILITY_AVG_PCT"].agg(
                lambda x: x.value_counts())  # .index[0])
            if mode_df.empty:
                lst_no_mode.append(item_id)
            else:
                mode = mode_df.index[0]
                print(f"{item_id}= {mode}")
                self.data.loc[(self.data["ITEM_ID"] == item_id) &
                              (self.data.VISIBILITY_AVG_PCT.isnull()), "VISIBILITY_AVG_PCT"] = mode

    def assign_demandpot(self):
        self.data["DEMAND_POT"] = self.data["DEMAND_PCS"] / self.data["VISIBILITY_AVG_PCT"]

    def check_inf_or_nan(self, col_lst=("DEMAND_PCS", "VISIBILITY_AVG_PCT", "DEMAND_POT")):
        # todo: Add checking for inf also
        """"
        Checking final frame for nans and infinity
        These columns ("DEMAND_PCS", "VISIBILITY_AVG_PCT", "DEMAND_POT") definitely should not have NAN
        """
        for col_name in col_lst:
            nans_in_col = self.data[col_name].isnull().sum()
            ic(nans_in_col)
            assert ( nans_in_col == 0), f"{col_name} should not have NAN values " \
                                        f"but {nans_in_col} found"







    def impute_missing_values(self):
        """ All the above functions shall be executed here"""
        if self.load_imputed_df:
            self.data = pd.read_pickle(self.load_imputed_df_name, compression="zip")
        else:
            self.choose_subset_data()
            #self.backup_original_columns()
            self.fill_missing_vis_by_mode() #todo: donot impute using mode
            self.fill_when_both_vis_pcs_nan(lower=self.lowest_visibility)
            self.fill_nan_pcs()
            self.clip_vis_and_pot(lower=self.lowest_visibility)
            self.origin_unknown_nans()
            self.assign_demandpot()
            self.check_inf_or_nan()

            return self.data
