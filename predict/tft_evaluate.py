import os
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from evaluation.evaluation_config import EvaluationConfig
from evaluation.metrics_evaluator import MetricsEvaluator
from evaluation.results_loader import ResultsLoader
import torch
from pytorch_forecasting.models.baseline import Baseline
import pdfrw
import glob
from pdfrw import PdfReader, PdfWriter
from matplotlib import pyplot as plt

from pytorch_forecasting.models import TemporalFusionTransformer
import statsmodels.api as sm
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import copy
from itertools import cycle
import logging
logger = logging.getLogger(__name__)


class EvaluateTFT:

    def __init__(self, data, train_test_config):
        self.data = data
        self.train_test_config = train_test_config
        for key, value in self.train_test_config.get_custom_param("predict_class").items():
            setattr(self, key, value)

        self.update_params_from_config()

    def update_params_from_config(self):
        self.max_prediction_length = self.train_test_config.get_custom_param("structuringdataset")[
            "max_prediction_length"]
        run_name = self.train_test_config.get_custom_param("run_name")
        path_tomake_dir_fig = f"{Path.cwd()}/All_data/{run_name}"
        Path(path_tomake_dir_fig + "/images").mkdir(exist_ok=True)
        Path(path_tomake_dir_fig + "/junk").mkdir(exist_ok=True)
        self.images_dir = str(Path(os.path.join(path_tomake_dir_fig, "images")))
        self.junk_dir = str(Path(os.path.join(path_tomake_dir_fig, "junk")))
        self.is_real_prediction = self.train_test_config.get_custom_param("is_real_prediction")
        self.run_pipeline_from_dataframe = self.train_test_config.get_custom_param("run_pipeline_from_dataframe")

    def log_all_prediction_to_mlflow(self, best_tft, val_dataloader, best_model_path):

        base_line_value = self.create_baseline_model(val_dataloader)
        mae_tft = self.get_mae_on_prediction(val_dataloader, best_tft)

        mlflow.log_metric("Baseline_value", base_line_value)
        mlflow.log_metric("MAE_TFT for val_load", mae_tft)
        mlflow.log_param("Best model path", best_model_path)
        mlflow.log_artifacts(self.images_dir)

        # mlflow.log_params(str(dict(best_tft.hparams)))

    def create_baseline_model(self, val_dataloader):
        # calculate baseline mean absolute error (MAE), i.e. predict next value as the last available value from the history
        actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
        baseline_predictions = Baseline().predict(val_dataloader)
        return ((actuals - baseline_predictions).abs().mean().item())

    def get_mae_on_prediction(self, val_dataloader, best_tft):
        # calcualte mean absolute error on validation set
        val_actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        val_predictions = best_tft.predict(val_dataloader, mode="prediction")
        mae_tft = ((val_actuals) - (val_predictions)).abs().mean().item()
        return mae_tft


    def get_legacy_actual_ann_data(self,  tmfrm_nm):
        """Evaluates results for the test dataset"""
        evaln_cnfg_path = Path(__file__).parent.parent.parent/ f'evaluation_config.yaml'
        config_ls = [evaln_cnfg_path]
        evaln_config = EvaluationConfig(config_ls)
        self.train_test_config.override_config(evaln_config._config_dict, self.train_test_config._config_dict)


        #If tmfrm_nm not in time frames of the evaluation config then pop the time frame key
        # from the dictionary
        lst_timeframes = list(evaln_config._config_dict["timeframes"].keys())
        for key in lst_timeframes:
            if key != tmfrm_nm:
                del evaln_config._config_dict["timeframes"][key]

        rslts_ldr = ResultsLoader(evaln_config)
        logger.info("Querying to get legacy, actuals, and ANN data")
        rslts_dict = rslts_ldr.load_forecast_results(tmfrm_nm, load_ai=True, load_actual=True,
                                                     load_legacy=True)

        return rslts_dict


    @staticmethod
    def get_itemids_only_modeltrained(best_model_path, df, assert_nonempty=False):
        """
        Extract only those ITEM_IDS that are present in the trained model
        and filter the df data based on those ITEM_IDS alone.
        """
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        # train_df = pd.read_pickle(logged_train_data_path, compression="zip")
        #
        # self.data = pd.concat([train_df, self.data], axis=0)

        # #todo: Check if dropping duplicates is required

        dict_trained_item_ids = dict(best_tft.hparams)["embedding_labels"]["ITEM_ID"]
        dict_trained_item_ids.pop("nan")
        trained_item_ids_df = pd.DataFrame(list(dict_trained_item_ids.items()),
                                           columns=['ITEM_ID', 'item_id_mapping'])
        df["ITEM_ID"] = df["ITEM_ID"].astype(str)
        df = pd.merge(trained_item_ids_df, df, how="inner", on=["ITEM_ID"])

        if assert_nonempty:
            assert len(df) > 0, "The trained item_ids and the obtained item_ids are different" \
                                   " and no common ids among them"
        return df

    def prepare_df_for_visualization(self, tmfrm_nm, best_model_path, prediction_df):
        """
        Make a single data frame consisting of predictions (along with their quantiles), actuals,
        ANN, tft, prophet so that the data frame itself can be used for plotting
        """

        rslts_dict = self.get_legacy_actual_ann_data(tmfrm_nm)
        dict_df = {}


        for key in rslts_dict.keys():

            # df_trained_itemids = EvaluateTFT.get_itemids_only_modeltrained(best_model_path,
            #                                                   rslts_dict[key])

            if not rslts_dict[key].empty():
                df_trained_itemids = pd.merge(df_trained_itemids, prediction_df, how="inner", on=["ITEM_ID","REFERENCE_YEAR","REFERENCE_WEEK"])
                dict_df[key] = df_trained_itemids

        #add the current tft so that model evaluator can give table and plots
        prediction_df.rename(columns={'RESULTS': 'FORECAST_PCS'}, inplace=True)
        dict_df["tft_current"] = prediction_df
        return dict_df

    def evaluate(self,tmfrm_nm, best_model_path, prediction_df):


        rslts_dict = self.prepare_df_for_visualization(tmfrm_nm, best_model_path, prediction_df)
        # TODO: Add comments for passing rslts_dict
        metrics_evaltr = MetricsEvaluator(rslts_dict, mlflow_logging=True, plots_enabled=True)
        metrics_dict = metrics_evaltr.evaluate_stats(
            min_visibility=self.train_test_config.get_custom_param('min_visibility'))

        del rslts_dict["tft_current"]
        return rslts_dict

    def visualize_individual_items(self, rslts_dict, best_tft, val_dataloader, new_pred_raw, new_x_raw,
                                   new_index_raw, new_index, new_x, new_pred):
        #todo: taken from jupyter notebook, need to understand what I wrote and refactor it

        def timeidx_2_date(lst_idx):
            lst_date = []
            for time_idx in lst_idx:
                date_index = timedelta(days=7 * time_idx) + self.data["FIRST_DAY_OF_TIMEFRAME"].min()
                lst_date.append(date_index)
            return lst_date

        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        pred_color = next(prop_cycle)["color"]

        listfig = []
        maelist = []
        actual_data = [(y[0]) for x, y in iter(val_dataloader)]

        for key in rslts_dict.keys():

            if key == "ACTUALS":
                rslts_dict[key]["VISIBILITY_AVG_PCT"].clip(lower=0.1, inplace=True)
                rslts_dict[key]["DEMAND_POT"] = rslts_dict[key]["DEMAND_PCS"] / rslts_dict[key][
                    "VISIBILITY_AVG_PCT"]

            elif key == "LEGACY":
               pass

            else:
                rslts_dict[key]["VISIBILITY_AVG_PCT"].clip(lower=0.1, inplace=True)
                rslts_dict[key]["DEMAND_POT"] = rslts_dict[key]["FORECAST_PCS"] / \
                                                rslts_dict[key]["VISIBILITY_AVG_PCT"]


        for i in list(new_index.index):

            pred_ts = new_pred[torch.tensor(i)].tolist()
            try:
                actuals = actual_data[0][i].tolist()
            except IndexError:
                pass

            # print(len(pred_ts))
            decoder_length1 = new_x["decoder_time_idx"][i].tolist()
            # print(len(decoder_length1))
            decoder_length1_date = timeidx_2_date(decoder_length1)
            subtracted = [abs(x1 - x2) for (x1, x2) in zip(actuals, pred_ts)]
            mae = np.array(subtracted).mean()

            # Preparation for plotting the Quantiles
            Quantiles_item = new_pred_raw[0][i, :, :]

            float_itemid = float(new_index["ITEM_ID"][i])
            print(f"{float_itemid}  = {mae}")

            # # limits in time series to get the forecast for ANN model
            # lst_prediction = new_x["decoder_time_idx"][0].tolist()
            # start = lst_prediction[0]
            # end = lst_prediction[-1]
            # print(f"start = {start} end = {end}")

            train_data = self.data.loc[self.data["ITEM_ID"] == str(float_itemid), ["FIRST_DAY_OF_TIMEFRAME", "DEMAND_POT"]]

            fig, ax = plt.subplots(figsize=(25, 16))

            # Create a colour code cycler e.g. 'C0', 'C1', etc.
            colour_codes = map('C{}'.format, cycle(range(10)))


            for key in rslts_dict.keys():

                if key == "ACTUALS":
                    actual_df = rslts_dict[key].loc[rslts_dict[key]["ITEM_ID"] == float_itemid]
                    ax.plot( actual_df["FIRST_DAY_OF_TIMEFRAME"],  actual_df['DEMAND_POT'],
                             linestyle='-.', marker='s', color='#55a000', label=key)

                elif key == "LEGACY":
                    legacy_df = rslts_dict[key].loc[rslts_dict[key]["ITEM_ID"] == float_itemid]
                    ax.plot(legacy_df["FIRST_DAY_OF_TIMEFRAME"],
                            legacy_df['FORECAST_PCS'],
                            linestyle='--', marker='s', color='#6CF949', label=key)

                else:
                    model_df = rslts_dict[key].loc[rslts_dict[key]["ITEM_ID"] == float_itemid]
                    ax.plot( model_df["FIRST_DAY_OF_TIMEFRAME"],
                             model_df['DEMAND_POT'],
                            linestyle='-.', marker='s', color=next(colour_codes), label=key)

            textstr = f"without visibility cut MAE = {mae}"
            ax.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top')

            plt.title(str(float_itemid), fontsize=25)

            ax.plot(train_data["FIRST_DAY_OF_TIMEFRAME"], train_data["DEMAND_POT"], label='ACTUAL',
                    marker="H", linestyle="-")
            ax.plot(decoder_length1_date, pred_ts, linestyle='--', marker='*', color='#ff7823',
                    label='TFT_FORECAST')


            for j in range(Quantiles_item.shape[1] // 2):
                ax.fill_between(decoder_length1_date, Quantiles_item[:, j],
                                Quantiles_item[:, -j - 1], alpha=0.15, fc=pred_color)

            ax.legend()
            # best_tft.plot_prediction(new_x_raw, new_pred_raw, idx=i, add_loss_to_title=True, ax=ax)

            fig.savefig(f"{self.junk_dir}/{i}.pdf", format="pdf")
            plt.close()

    def merge_to_single_pdf(self):

        writer = PdfWriter()
        for inpfn in glob.glob(f"{self.junk_dir}/*.pdf"):
            writer.addpages(PdfReader(inpfn).pages)
        writer.write(self.output_filename)

        #todo:copy the output file to previous directory and delete the junk directory


    def qqplot(self, result_tft_df):
        # todo: save the plots
        # sm.qqplot_2samples(merged_df["DEMAND_POT"], merged_df["DFC_DEMAND_POT"],
        #                    xlabel="DEMAND_POT", ylabel="DFC_DEMAND_POT", line="45")
        # sm.qqplot_2samples(merged_df["DFC_DEMAND_POT"], merged_df["Predictions_tft"],
        #                    xlabel="DFC_DEMAND_POT", ylabel="TFT", line="45")
        fig = sm.qqplot_2samples(result_tft_df["RESULT"], result_tft_df["ACTUALS"],
                                 xlabel="RESULT_TFT", ylabel="ACTUALS", line="45")

        fig.savefig(f"{self.images_dir}/qqplot_Actuals_vs_TFT.png")

    def get_production_data_and_compare(self, df_tft):
        results_dict = self.get_legacy_actual_ann_data()

        ann_forecast = self.data[
            ["time_idx", "DFC_DEMAND_POT", "DEMAND_POT", "ITEM_ID", "FIRST_DAY_OF_TIMEFRAME"]]

        merged_df = pd.merge(ann_forecast, df_tft, how="inner", on=["time_idx", "ITEM_ID"])

        mae_actual_vs_tft = mean_absolute_error(merged_df["DEMAND_POT"],
                                                merged_df["Predictions_tft"])

        mae_actual_vs_ann = mean_absolute_error(merged_df["DEMAND_POT"],
                                                merged_df["DFC_DEMAND_POT"])

        print(f"mae_actual_vs_tft = {mae_actual_vs_tft}\n")
        print(f"mae_actual_vs_ann = {mae_actual_vs_ann}\n")

