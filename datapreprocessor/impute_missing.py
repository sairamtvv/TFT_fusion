import pandas as pd
import numpy as np


class ImputeMissingValues:

    def __init__(self, config):
        self.config = config


        # Parameters imported from YAML file

        for key, value in self.config.get_custom_param("imputemissingvalues").items():
            setattr(self, key, value)

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
