from pathlib import Path
import os
import pa_core
import copy

from evaluation.evaluation_config import EvaluationConfig
from evaluation.metrics_evaluator import MetricsEvaluator
from evaluation.results_loader import ResultsLoader
from pa_demand_forecast.raw_forecast_tft.demand_forecast_TFT import TFTDemandForecast

DFLT_CFGDIR = 'default_configs'
def get_trntest_pipeline_config(experiment_name):
    exp_config = Path(os.getcwd()).parent.parent / f'pipeline_config_{experiment_name}.yaml'
    config_ls = [
        Path(os.getcwd()).parent.parent/ DFLT_CFGDIR / 'forecast_config_TFT.yaml',
        Path(os.getcwd()).parent.parent / DFLT_CFGDIR / 'feature_engineer_config.yaml',
        exp_config
    ]
    pipeline_config = pa_core.config.TrainTestPipelineConfig(config_ls)
    return pipeline_config


def evaluate(actual_df, tmfrm_nm, train_test_config):
    """Evaluates results for the test dataset"""
    evaln_cnfg_path = Path(__file__).parent / f'evaluation_config.yaml'
    config_ls = [evaln_cnfg_path]
    evaln_config = EvaluationConfig(config_ls)
    train_test_config.override_config(evaln_config._config_dict,
                                           train_test_config._config_dict)

    # If tmfrm_nm not in time frames of the evaluation config then pop the time frame key from the dictionary
    lst_timeframes = list(evaln_config._config_dict["timeframes"].keys())
    for key in lst_timeframes:
        if key != tmfrm_nm:
            del evaln_config._config_dict["timeframes"][key]

    rslts_ldr = ResultsLoader(evaln_config)
    rslts_dict = rslts_ldr.load_forecast_results(tmfrm_nm, load_ai=False, load_actual=True,
                                                 load_legacy=False)
    #rslts_ldr.close_db_connections()
    # experiment_name = self.train_test_config.get_custom_param("mlflow_experiment_name")
    # metrics_evaltr = MetricsEvaluator(rslts_dict, mlflow_exp_id=mlflow.get_experiment_by_name(
    #     experiment_name).experiment_id)
    # metrics_dict = metrics_evaltr.evaluate_stats(
    #     min_visibility=self.train_test_config.get_custom_param('min_visibility'))

if __name__ == "__main__":
    prediction_config = pa_core.config.PredictPipelineConfig(
        ["../../prediction_config.yaml", "../../prediction_config_TFT.yaml"])
    pipeline_config = get_trntest_pipeline_config("TFT")

    pipeline_config_copy = copy.deepcopy(pipeline_config)
    pipeline_config.override_config(pipeline_config._config_dict, prediction_config._config_dict)

    pipeline_config._config_dict["regions"] = pipeline_config_copy._config_dict["regions"]

    train_test_config = pipeline_config

    tmfrm_nm = "real_prediction_run_1"
    evaluate(None,tmfrm_nm,train_test_config)