import os

import mlflow
from pa_core.config import TrainTestPipelineConfig
from pa_core.model.model import BaseModel

from model_builder_trainer.optuna_tuning import OptunaTuning
from model_builder_trainer.tft_model_definition import TFTModelDefinition
from model_builder_trainer.structure_data import \
    StructuringDataset
from predict.predict import Predict

from pathlib import Path



import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class TFTDFCModel():

    def __init__(self, data, train_test_config):
        """Builds the model, finds learning rate also trains  the model"""

        self.data = data
        self.train_test_config = train_test_config

        for key, value in self.train_test_config.get_custom_param("tftdfcmodel").items():
            setattr(self, key, value)

    def prepare_data(self, predict):
        structure_data_obj = StructuringDataset(self.data, self.train_test_config)
        training, training_cutoff = structure_data_obj.structure_as_timeseries_dataset()
        train_dataloader, val_dataloader = structure_data_obj.prepare_dataloader(training,
                                                                                 training_cutoff,
                                                                                 self.data,
                                                                                 predict=predict)
        return structure_data_obj, training, training_cutoff, train_dataloader, val_dataloader

    def train(self):

        structure_data_obj, training, training_cutoff, train_dataloader, val_dataloader = self.prepare_data(predict=False)
        model_defn_obj = TFTModelDefinition(self.data, self.train_test_config)
        trainer, tft_model = model_defn_obj.configure_tft_network(training, run_type=self.run_type)
        model_defn_obj.log_all_parameters_to_mlflow(structure_data_obj, training, tft_model)
        # fit network
        if self.run_type == "find_lr":
            self.find_learning_rate(trainer, tft_model, train_dataloader, val_dataloader)
            exit()

        elif self.run_type == "train":
            trainer.fit(tft_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            best_model_path = trainer.checkpoint_callback.best_model_path

            #todo: hard coded path sync with demand_forecast.py
            #logging the model checkpoint as artifact in mlflow
            tensorboard_log_dir = f"{Path(os.getcwd())}/All_data/" + self.train_test_config.get_custom_param("run_name")
            lightning_dir = f'{tensorboard_log_dir}/lightning_logs/'
            mlflow.log_artifacts(lightning_dir)

            self.predict(best_model_path)

            return best_model_path

        elif self.run_type == "optuna_tune":
            optuna_tuning_obj = OptunaTuning(self.data)
            optuna_tuning_obj.tft_optuna_tune(train_dataloader, val_dataloader)
            exit()

        else:
            raise ValueError("Train can only perform find_learning_rate, train or optuna_tuning")

    def find_learning_rate(self,trainer, tft_model, train_dataloader, val_dataloader):
         # find optimal learning rate
         res = trainer.tuner.lr_find(
             tft_model,
             train_dataloaders=train_dataloader,
             val_dataloaders=val_dataloader,
             max_lr=10.0,
             min_lr=1e-6,
         )

         print(f"suggested learning rate: {res.suggestion()}")
         fig = res.plot(show=True, suggest=True)
         fig.show()




    def predict(self, best_model_path):


        predict_obj = Predict(self.data, self.train_test_config)
        result_df = predict_obj.predict(best_model_path)

        return result_df




    def default_model_definition(self):
        tftmodel_definition = TFTModelDefinition()
        return tftmodel_definition


# todo: save model_builder_trainer after every epoch after 2 or 3 epochs and check which model_executor is the best
# todo: check the model_executor by doing a group by on a product line or even a style id incase we have that. Check whether that improves MAE
# todo: adding logging
# todo: enable and disable icecream from the configfile
# todo: add list of todos from file
# todo:Plot the items not in the validation loader
#
# todo: Check which run is better, one with the complete dataset or
# todo:good to have scheduler for different configurations of TFT
# todo:Please dont forget to remove those items which has no shadow. The model_executor couldnot predict quantiles. Those should be removed from the prediction, Probbaly they are NANS
# todo: Should there be sperate runs for one:good amount of time series ITEMS and another over complete S items
# todo: Now that outliers are removed, probably I can try standard scaling can yield better results
# todo: Add Logger
# todo: Add MLFLOW
#
# todo: donot forget to deallocate RAM where the consumption is maximum
# todo: QQ PLOT UNDERSTANDING
#
# todo: for the second run in the list RAM allocated for the first run is deallocated or not
# todo: Currently, loads data for every run, should that be the case or should data be loaded only once, get that functionality also


# todo: if the date is greater than today, then make visibility= 1 but today can keep changing depending on the availability of the data , this should be implemented
#todo: write a program to generate the training_cutoff number required in the validation
#todo: Make a column of visibility_new, ideally the model should understand that the demand_pcs reduce with respect to visibility but the demand pot becomes double..(in my case)
#todo: trying huber loss by controlling delta and omega to account for outliers
#todo: Quantile what range should be used 80% or 98%
#todo: The idea is to predict the demand_pcs, (well behaved function),

#todo: Adjust the right reference date, get test data completely different from  the train data frame

