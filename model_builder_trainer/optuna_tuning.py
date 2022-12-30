import pickle
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


class OptunaTuning:

    def __init__(self, config):
        for key, value in self.get_yaml_params().items():
            setattr(self, key, value)

        self.optuna_best_model_save_dir = "lightning_logs/"+ self.optuna_best_model_save_dir
        self.optuna_save_study_filename = self.optuna_best_model_save_dir +"/" + self.optuna_save_study_filename +".pkl"

    def get_yaml_params(self):
        exp_config = f"../../pipeline_config_{self.config}.yaml"
        pipeline_config = TrainTestPipelineConfig(exp_config)
        dict_yamlparams = pipeline_config.get_custom_param("optunatuning")
        return dict_yamlparams

    def tft_optuna_tune(self, train_dataloader, val_dataloader):
        # create study
        #todo: implement the ranges in the yaml file
        #todo: implement if study file exists, to reuse the exisiting study file
        study = optimize_hyperparameters(train_dataloader,
                                         val_dataloader,
                                         model_path=self.optuna_best_model_save_dir,
                                         n_trials=100,
                                         max_epochs=1,
                                         gradient_clip_val_range=(0.01, 0.6),
                                         hidden_size_range=(8, 64),
                                         # max_encoder_length_range = (6,7),
                                         hidden_continuous_size_range=(3, 32),
                                         attention_head_size_range=(1, 4),
                                         learning_rate_range=(0.00001, 0.001),
                                         dropout_range=(0.1, 0.5),
                                         # trainer_kwargs=dict(limit_train_batches=30),
                                         reduce_on_plateau_patience=4,
                                         # study = study_file,
                                         # weight_decay= (0.001, 0.1),
                                         verbose=True,
                                         use_learning_rate_finder=False,
                                         # use Optuna to find ideal learning rate or use in-built learning rate finder
                                         )

        # save study results - also we can resume tuning at a later point in time
        with open(self.optuna_save_study_filename, "wb") as fout:
            pickle.dump(study, fout)

        # show best hyperparameters
        print(study.best_trial.params)
        # {'gradient_clip_val': 0.7730088999177774, 'hidden_size': 16, 'dropout': 0.4310838003963967, 'hidden_continuous_size': 1, 'attention_head_size': 2, 'learning_rate': 0.009255580254316748}


