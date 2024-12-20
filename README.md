# 02456-Deep-Learning-Project
Codebase for results of the final project in the Deep Learning course 02456 Fall 2024

## Pre-requisites
To produce the hydrograph plot you need the original data which is confidential, however using a dataset which follows the neural hydrology framework's specifications, you should be able to produce your own.

You need Neural Hydrology.

You need to download US CAMELS dataset from [US CAMELS](https://ral.ucar.edu/solutions/products/camels)

You need to download DK dataset from [DK dataset](https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/AZXSYP) \
This is the DK CAMELS dataset, NOT the DK subset CAMELS dataset which was used to produce the results in the paper.

## Run configs for model training
Located in the run_config and run_config_us_camels folders are the configuration files used to train the models. 

## Python files
``_model_run.py`` contains the classes used to hold the model test results and computes the metrics.

``_tools.py`` contains most of the paths used in the other files and contains the root path. Change this root path to the project folder on your local machine.

``_train_test_eval.py`` contains the code for training the models and evaluating the results.

``data_splitting.py`` contains the code for splitting the data into training, validation and test sets. For the DK dataset the data is split into 70% training, 15% validation and 15% test. The data is split per basin, so the same basin is not present in both the training and test set.

``evaluation_comparison.py`` contains the code for comparing the results of the different models and the code to reproduce the plots.

In order to run training, the neural hydrology framework should be downloaded and installed following the guide: [NeuralHydrology](https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html#installation)

## Folder structure
This is the folder structure of the project

```
FOLDER WHICH README.md RESIDES IN
...
+---data_config
+   +---per_basin
+   +---per_basin_qsim
+   +---per_basin_qsim_2002
+   +---per_basin_qsim_2011
+   +---us_camels
+---DK_SUBSET_CAMELS
+   +---attributes
+   +---time_series
+---figures
+---generated
+   +---data_config
+       +---per_basin
+       +---per_basin_qsim
+       +---per_basin_qsim_2011
+---neuralhydrology
+   +---datasetzoo
...
+---neuralhydrology.egg-info
+---runs
+   +---nse_dk_lstm_attributes_per_basin_365_1810_112154
+   +   +---img_log
+   +   +---test
+   +   +   +---model_epoch030
+   +   +---train
+   +   +   +---model_epoch030
+   +   +---train_data
+   +---nse_us_lstm_attributes_365_1111_113602
+   +   +---img_log
+   +   +---test
+   +   +   +---model_epoch030
+   +   +---train_data
+---run_configs
+   +---percent_split
+   +   +---MSE
+   +   +---NSE
+   +---per_basin
+       +---MSE
+       +   +---seq_length_30
+       +   +---seq_length_365
+       +   +---seq_length_90
+       +---NSE
+           +---qsim_2002
+           +---qsim_2011
+           +---seq_length_30
+           +---seq_length_365
+           +---seq_length_90
+---run_configs_us_camels
+---US_CAMELS
+   +---basin_timeseries_v1p2_metForcing_obsFlow
...
```