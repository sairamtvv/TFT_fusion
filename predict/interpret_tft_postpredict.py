import numpy as np
import os
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
class InterpretTFTPostPredict:
    def __init__(self, data, train_test_config):
        self.data = data
        self.train_test_config = train_test_config
        self.update_params_from_config()

    def update_params_from_config(self):
        self.max_prediction_length = self.train_test_config.get_custom_param("structuringdataset")[
            "max_prediction_length"]
        run_name = self.train_test_config.get_custom_param("run_name")
        path_tomake_dir_fig = f"{Path.cwd()}/All_data/{run_name}"
        Path(path_tomake_dir_fig + "/images").mkdir(exist_ok=True)
        self.images_dir = str(Path(os.path.join(path_tomake_dir_fig, "images")))
        self.is_real_prediction = self.train_test_config.get_custom_param("is_real_prediction")
        self.run_pipeline_from_dataframe = self.train_test_config.get_custom_param("run_pipeline_from_dataframe")

    def interpret_model(self, best_tft, raw_predictions):

        interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
        dict_mtpltlib_fig = best_tft.plot_interpretation(interpretation)

        for name, fig in dict_mtpltlib_fig.items():
            fig
            fig.savefig(f"{self.images_dir}/{name}.png")

    def partial_dependency(self, best_tft, val_dataloader, filter_item_id=None, lst_input_feat=None):
        """
        Takes as an input list of input_features, calculates their upper and lower bounds,
        extract their partial dependency and save their partial dependency plots
        """
        if lst_input_feat is None:
            lst_input_feat = ["SYSTEM"] \
                             + ['DOF_AVG',
                              'DIFUSION_COEFFICIENT', 'SCREENING_POTENTIAL', 'REPRODUCIBILITY',
                              'DISCREPANCY', 'MEASUREMENT_FLAW'] \

                              # + ['SYSTEM', 'YEAR',  'time_idx', 'METRIC'] \
                             # + ['avg_QUANTILE_NORM_by_SYSTEM', 'QUANTILE_NORM_lagged_1',
                             #   'QUANTILE_NORM_lagged_2'] \
                             # + ['NEW_RESEARCHERS', 'CUM_NO_RESEARCHERS',  ]


        for input_feature in lst_input_feat:
            if self.data[input_feature].dtype == "category":
                upper_bound_feature = self.data[input_feature].nunique()
                lower_bound_feature = 0
                feature_range = np.arange(lower_bound_feature, upper_bound_feature, 1)

            else:
                upper_bound_feature = self.data[input_feature].max()
                lower_bound_feature = self.data[input_feature].min()
                feature_range = np.linspace(lower_bound_feature, upper_bound_feature, 100)

            dependency = best_tft.predict_dependency(val_dataloader.dataset,
                                                     input_feature, feature_range,
                                                     show_progress_bar=True,
                                                     mode="dataframe")

            # plotting median and 25% and 75% percentile
            agg_dependency = dependency.groupby(input_feature).prediction.agg(
                median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
            )
            ax = agg_dependency.plot(y="median")
            ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3)
            ax.figure.savefig(f"{self.images_dir}/{input_feature}_partial.png")
            #123

