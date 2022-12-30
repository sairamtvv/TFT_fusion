import os

import pa_core
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pa_core
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting.models.baseline import Baseline
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MQF2DistributionLoss, \
    PoissonLoss, TweedieLoss, MAE, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from torchmetrics import MeanAbsoluteError
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from pathlib import Path
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

import pdfrw
import glob
from pdfrw import PdfReader, PdfWriter

from evaluation.evaluation_config import EvaluationConfig
from evaluation.metrics_evaluator import MetricsEvaluator
from evaluation.results_loader import ResultsLoader
from pa_demand_forecast.raw_forecast_tft.model_builder_trainer.structure_data import \
    StructuringDataset
import mlflow
from datetime import datetime, timedelta


class Predict:
    # todo: plot the actual histogram of demand_pot over the tft_histogram of demand_pot

    def __init__(self, data, train_test_config):
        self.data = data
        self.train_test_config = train_test_config

        # todo: can make this elegant
        # these custom parameters needs to be present in both train and test config. But only predict config is used
        for key, value in self.train_test_config.get_custom_param("predict_class").items():
            setattr(self, key, value)
        self.update_params_from_config()

        #
        #
        # self.model = model
        # self.input = input1
        # self.loss_fn = loss_fn
        # self.optimizer = optimizer
        # self.metric = metric
        # self.epoches = epoches
        #
        # self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        # self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './tf_ckpts', max_to_keep=3)
        #
        # self.train_log_dir = 'logs/gradient_tape/'
        # self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        #
        # self.model_save_path = 'saved_models/'

    def update_params_from_config(self):
        self.max_prediction_length = self.train_test_config.get_custom_param("structuringdataset")[
            "max_prediction_length"]
        run_name = self.train_test_config.get_custom_param("run_name")
        path_tomake_dir_fig = f"{Path.cwd()}/All_data/{run_name}"
        Path(path_tomake_dir_fig + "/images").mkdir(exist_ok=True)
        self.images_dir = str(Path(os.path.join(path_tomake_dir_fig, "images")))
        self.is_real_prediction = self.train_test_config.get_custom_param("is_real_prediction")
        self.run_pipeline_from_dataframe = self.train_test_config.get_custom_param("run_pipeline_from_dataframe")


    def prepare_data(self, predict):
        structure_data_obj = StructuringDataset(self.data, self.train_test_config)
        training, training_cutoff = structure_data_obj.structure_as_timeseries_dataset()
        train_dataloader, val_dataloader = structure_data_obj.prepare_dataloader(training,
                                                                                 training_cutoff,
                                                                                 self.data,
                                                                                 predict=predict)
        return structure_data_obj, training, training_cutoff, train_dataloader, val_dataloader

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

    def predict(self, best_model_path, fcst_tmfrm_nm=None):
        structure_data_obj, training, training_cutoff, train_dataloader, \
            val_dataloader = self.prepare_data(predict=True)

        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        new_pred_raw, new_x_raw, new_index_raw = best_tft.predict(val_dataloader, mode="raw",
                                                                  return_x=True, return_index=True)

        # todo: Add quantiles to the result_df
        # todo: Making the actual train_df and merging with the incoming test data
        # todo: Adding individual plots making ANN as a input parameter
        # todo: Doing the correct preprocessing for the target data
        #todo: Write the code for data not in validation data set
        #todo: actuals_vs_predictions plot, make a directory in images and save
        #todo: currently time_idx has a standard scalar which is very wrong, remove it


        result_tft_df = self.make_tft_predictions_df(best_tft, val_dataloader)
        result_tft_df = self.get_desired_df_for_gross_forecast(result_tft_df)

        # interpretation
        self.interpret_model(best_tft, new_pred_raw)
        #self.partial_dependency(best_tft, val_dataloader, lst_input_feat=None)



        # # comparision with the other models
        # if actual_df is None:
        #     # self.get_production_data_and_compare(df_tft)
        #     self.evaluate(actual_df, fcst_tmfrm_nm)
        #     self.qqplot(result_tft_df)




        self.log_all_prediction_to_mlflow(best_tft, val_dataloader, best_model_path)

        return result_tft_df

    def evaluate(self, actual_df, tmfrm_nm):
        """Evaluates results for the test dataset"""
        evaln_cnfg_path = Path(__file__).parent / f'evaluation_config.yaml'
        config_ls = [evaln_cnfg_path]
        evaln_config = EvaluationConfig(config_ls)
        self.train_test_config.override_config(evaln_config._config_dict, self.train_test_config._config_dict)

        #If tmfrm_nm not in time frames of the evaluation config then pop the time frame key from the dictionary
        for key in evaln_config._config_dict["timeframes"].keys():
            if key != tmfrm_nm:
                del evaln_config._config_dict["timeframes"][key]

        rslts_ldr = ResultsLoader(evaln_config)
        rslts_dict = rslts_ldr.load_forecast_results(tmfrm_nm, load_ai=False, load_actual=True, load_legacy=True)
        rslts_ldr.close_db_connections()
        experiment_name = self.train_test_config.get_custom_param("mlflow_experiment_name")
        metrics_evaltr = MetricsEvaluator(rslts_dict, mlflow_exp_id=mlflow.get_experiment_by_name(experiment_name).experiment_id)
        metrics_dict = metrics_evaltr.evaluate_stats(min_visibility=self.train_test_config.get_custom_param('min_visibility'))



    def interpret_model(self, best_tft, raw_predictions):
        interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
        dict_mtpltlib_fig = best_tft.plot_interpretation(interpretation)
        for name, fig in dict_mtpltlib_fig.items():
            fig
            fig.savefig(f"{self.images_dir}/{name}.png")

    def qqplot(self, result_tft_df):
        # todo: save the plots
        # sm.qqplot_2samples(merged_df["DEMAND_POT"], merged_df["DFC_DEMAND_POT"],
        #                    xlabel="DEMAND_POT", ylabel="DFC_DEMAND_POT", line="45")
        # sm.qqplot_2samples(merged_df["DFC_DEMAND_POT"], merged_df["Predictions_tft"],
        #                    xlabel="DFC_DEMAND_POT", ylabel="TFT", line="45")
        fig = sm.qqplot_2samples(result_tft_df["RESULT"], result_tft_df["ACTUALS"],
                                 xlabel="RESULT_TFT", ylabel="ACTUALS", line="45")

        fig.savefig(f"{self.images_dir}/qqplot_Actuals_vs_TFT.png")

    def get_df_for_forecast(self, best_model_path):
        """
        Extract only those ITEM_IDS that are present in the trained model
        and filter the train+prediction data based on those ITEM_IDS alone.
        """
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        # train_df = pd.read_pickle(logged_train_data_path, compression="zip")
        #
        # self.data = pd.concat([train_df, self.data], axis=0)

        # #todo: Check if dropping duplicates is required

        dict_trained_item_ids = dict(best_tft.hparams)["embedding_labels"]["ITEM_ID"]
        dict_trained_item_ids.pop("nan")
        trained_item_ids_df = pd.DataFrame(list(dict_trained_item_ids.items()), columns=['ITEM_ID','item_id_mapping'])
        self.data["ITEM_ID"] = self.data["ITEM_ID"].astype(str)
        self.data = pd.merge(trained_item_ids_df, self.data, how="inner", on=["ITEM_ID"])

        assert len(self.data) > 0, "The trained item_ids and the prediction item_ids are different and no common ids among them"
        self.data = self.data.loc[self.data["time_idx"]<=234, self.data.columns]


    def predict_evaluate_with_new_data(self, fcst_tmfrm_nm=None):
        """Read the old data and add the new incoming data """
        # read the old train_data from the data frame



        # Using Mlflow to load the best_model_path and train_data path saved inside the MLFLOW experiment

        logged_checkpoint_path = f"./mlruns/{self.toread_mlflow_experiment_id}/" \
                            f"{self.toread_mlflow_run_id}/artifacts/lightning_logs/version_0/checkpoints/" \
                            f"{self.toread_mlflow_ckpt_filename}"

        # logged_train_data_path = f"./mlruns/{self.toread_mlflow_experiment_id}/" \
        #                     f"{self.toread_mlflow_run_id}/artifacts/" \
        #                     f"{self.toread_mlflow_train_filename}"

        self.get_df_for_forecast(logged_checkpoint_path)


        # #make a outer join with the new data
        # #self.data = pd.merge(old_train_df, self.data, how="outer", on=["time_idx", "ITEM_ID"])
        # self.data = self.get_only_train_data_from_df()
        result_df = self.predict(logged_checkpoint_path, fcst_tmfrm_nm)



        return result_df

    def get_desired_df_for_gross_forecast(self, result_df):

        self.data["time_idx"] = (round(((self.data["FIRST_DAY_OF_TIMEFRAME"] - self.data[
            "FIRST_DAY_OF_TIMEFRAME"].min()).dt.days) / 7)).astype(int)
        self.data["ITEM_ID"] = self.data["ITEM_ID"].astype(str).astype('float32')

        required_cols = ["OBJECT_ID", "REGION_ID", "REGION_ALIAS", "ITEM_ID", "PRODUCTLINE",
                         "TIMEFRAME_ID", "RESULT", "MP_ENABLED_ISO"]


        result_df = pd.merge(result_df, self.data, how="inner", on=["ITEM_ID", "time_idx"])

        return result_df

    def make_tft_predictions_df(self, best_tft, val_dataloader):
        new_pred, new_x, new_index = best_tft.predict(val_dataloader, mode="prediction",
                                                      return_x=True, return_index=True)
        list1 = []
        maelist = []
        actual_data = [(y[0]) for x, y in iter(val_dataloader)]
        for i in list(new_index.index):
            pred_ts = new_pred[torch.tensor(i)].tolist()
            try:
                actuals = actual_data[0][i].tolist()
            except IndexError:
                pass
            # print(len(pred_ts))
            decoder_length1 = new_x["decoder_time_idx"][i].tolist()
            # print(len(decoder_length1))
            ITEM_ID = []
            ITEM_ID.append(new_index["ITEM_ID"][i])
            ITEM_ID = ITEM_ID * self.max_prediction_length
            # *new_x["decoder_time_idx"].size()[1] # multipying by  26
            # print(len(ITEM_ID))
            zipped = list(zip(actuals, pred_ts, decoder_length1, ITEM_ID))
            subtracted = [abs(x1 - x2) for (x1, x2) in zip(actuals, pred_ts)]
            mae = np.array(subtracted).mean()
            maelist.append(mae)
            print(f"{ITEM_ID[0]}  = {mae}")
            # print(list(zipped))

            [list1.append(item) for item in zipped]

            df_tft = pd.DataFrame(list1, columns=["ACTUALS", 'RESULT', 'time_idx', "ITEM_ID"])
            df_tft["ITEM_ID"] = df_tft["ITEM_ID"].astype(str).astype("float32")
            # RAW_COLUMN_LIST = ["PAI_REGION_ID","REGION_ALIAS","ITEM_ID","OBJECT_ID","PAI_TIMEFRAME_ID","RESULT","MP_ENABLED","PRODUCTLINE"]
        return df_tft

    def get_production_data_and_compare(self, df_tft):
        ann_forecast = self.data[
            ["time_idx", "DFC_DEMAND_POT", "DEMAND_POT", "ITEM_ID", "FIRST_DAY_OF_TIMEFRAME"]]

        merged_df = pd.merge(ann_forecast, df_tft, how="inner", on=["time_idx", "ITEM_ID"])

        mae_actual_vs_tft = mean_absolute_error(merged_df["DEMAND_POT"],
                                                merged_df["Predictions_tft"])

        mae_actual_vs_ann = mean_absolute_error(merged_df["DEMAND_POT"],
                                                merged_df["DFC_DEMAND_POT"])

        print(f"mae_actual_vs_tft = {mae_actual_vs_tft}\n")
        print(f"mae_actual_vs_ann = {mae_actual_vs_ann}\n")

    def visualize_individual_items(self, best_tft, val_dataloader, new_pred_raw, new_x_raw,
                                   new_index_raw):

        def timeidx_2_date(lst_idx):
            lst_date = []
            for i in lst_idx:
                date_i = timedelta(days=7 * i) + data["FIRST_DAY_OF_TIMEFRAME"].min()
                lst_date.append(date_i)
            return lst_date

        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        pred_color = next(prop_cycle)["color"]

        listfig = []
        maelist = []
        actual_data = [(y[0]) for x, y in iter(val_dataloader)]

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

            # limits in time series to get the forecast for ANN model
            lst_prediction = new_x["decoder_time_idx"][0].tolist()
            start = lst_prediction[0]
            end = lst_prediction[-1]
            print(f"start = {start} end = {end}")

            train_data = df.loc[
                df["ITEM_ID"] == str(float_itemid), ["FIRST_DAY_OF_TIMEFRAME", "DEMAND_POT"]]

            id_dfc_forecast = ann_forecast.where(
                (ann_forecast["ITEM_ID"] == float_itemid) & (ann_forecast["time_idx"] >= start) & (
                        ann_forecast["time_idx"] <= end)).dropna(how="all").sort_values(
                "time_idx")

            fig, ax = plt.subplots(figsize=(25, 16))
            textstr = f"without visibility cut MAE = {mae}"
            ax.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top')

            plt.title(str(float_itemid), fontsize=25)

            ax.plot(train_data["FIRST_DAY_OF_TIMEFRAME"], train_data["DEMAND_POT"], label='ACTUAL',
                    marker="H", linestyle="-")
            ax.plot(decoder_length1_date, pred_ts, linestyle='--', marker='*', color='#ff7823',
                    label='TFT_FORECAST')
            ax.plot(id_dfc_forecast["FIRST_DAY_OF_TIMEFRAME"], id_dfc_forecast['DFC_DEMAND_POT'],
                    linestyle='-.', marker='s', color='#55a000', label='DFC_FORECAST')

            for j in range(Quantiles_item.shape[1] // 2):
                ax.fill_between(decoder_length1_date, Quantiles_item[:, j],
                                Quantiles_item[:, -j - 1], alpha=0.15, fc=pred_color)

            ax.legend()
            # best_tft.plot_prediction(new_x_raw, new_pred_raw, idx=i, add_loss_to_title=True, ax=ax)
            fig.savefig("junk/" + str(i) + ".pdf", format="pdf")
            plt.close()

    def merge_to_single_pdf(self):
        outfn = "52weeks_prediction.pdf"
        writer = PdfWriter()
        for inpfn in glob.glob("*.pdf"):
            writer.addpages(PdfReader(inpfn).pages)
        writer.write(outfn)

    def partial_dependency(self, best_tft, val_dataloader, filter_item_id=None, lst_input_feat=None):
        """
        Takes as an input list of input_features, calculates their upper and lower bounds,
        extract their partial dependency and save their partial dependency plots
        """
        if lst_input_feat is None:
            lst_input_feat = ["time_idx", "VISIBILITY_AVG_PCT", "MP_ENABLED_ISO",
                              "CUM_CAT_SP_TEUR", "REFERENCE_WEEK", "REFERENCE_MONTH",
                              "COLLECTION", "FTOT", "EC_SP_TEUR", "SEASONALITY",
                              "WEEK_OF_MONTH"]


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
            agg_dependency = dependency.groupby(input_feature).normalized_prediction.agg(
                median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
            )
            ax = agg_dependency.plot(y="median")
            ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3)
            ax.figure.savefig(f"{self.images_dir}/{input_feature}_partial.png")

        # todo: pickling var_collection to reload again
        # todo: lagged visibility
        # todo: reference date is important for back testing
