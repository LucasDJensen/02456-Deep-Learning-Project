from datetime import datetime
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from os.path import join as pj

from _metrics_calculator import *


class ModelRun:
    label:str
    model_run_name:str
    epoch_folder:str
    
    metrics:pd.DataFrame
    
    all_basins:list[str]
    filtered_basins:list[str]
    no_observations_basins:list[str]

    nc_file_dataset:dict[str, xr.Dataset]
    qobs_label:str
    qsim_label:str

    def __init__(self, model_run_name:str, qobs_label:str, qsim_label:str):
        self.model_run_name = model_run_name
        self.qobs_label = qobs_label
        self.qsim_label = qsim_label

        self.no_observations_basins = []
        self.all_basins = []
        self.filtered_basins = []
        self.nc_file_dataset = {}

        self.label = self.make_label()

    def load_results(self):
        pass

    def get_qobs_qsim(self, basin:str):
        qobs = self.nc_file_dataset[basin][self.qobs_label].values.flatten()
        qsim = self.nc_file_dataset[basin][self.qsim_label].values.flatten()
        
        qsim, qobs = drop_missing_obs_or_sim_values(qsim, qobs)

        return qobs, qsim

    def make_label(self):
        label = []
        
        for part in self.model_run_name.split('_')[:-2]:
            if part == "attributes":
                label.append("ATTR")
            elif part == "dk":
                continue
            elif part.isnumeric():
                label.append(part)
            else:
                label.append(part.upper())
        return ' '.join(label)
    
    
    def calculate_basin_metrics(self):
        values = []
        for basin in self.filtered_basins:
            qsim, qobs = self.get_qobs_qsim(basin)
            
            fbal = fbal_calc(qsim, qobs)
            critical_success_index, n_hits, n_false, n_misses = critical_success_index_over_threshold(qsim, qobs, percentile=0.01, threshold_value=0, tolerance=0.1)
            mse = mse_calc(qsim, qobs)
            nse = nse_calc(qsim, qobs)
            kge = kge_calc(qsim, qobs)

            if np.isnan(critical_success_index):
                critical_success_index, n_hits, n_false, n_misses = -np.inf, -np.inf, -np.inf, -np.inf
        
            values.append([basin, fbal, critical_success_index, mse, kge, nse, n_hits, n_false, n_misses])

        df = pd.DataFrame(values, columns=['basin','FBAL', 'CSI', 'MSE', 'KGE', 'NSE', 'n_hits', 'n_false', 'n_misses'])
        # df.dropna(inplace=True)

        self.metrics = df

    def filter_basins(self, basins:list[str]):
        self.filtered_basins = basins.copy()

    def get_metrics(self):
        return self.metrics[self.metrics['basin'].isin(self.filtered_basins)]
    
    def plot_simulated(self, ax:plt.Axes, basin:str):
        qobs, qsim = self.get_qobs_qsim(basin)
        dates = self.nc_file_dataset[basin]['date']
        ax.plot(dates, qsim, label=f'{self.label} - {len(self.filtered_basins)}')

    def quality(self, basin:str):
        qobs, qsim = self.get_qobs_qsim(basin)
        return len(qsim) / len(qobs)
    

class DKHYPE_ModelRun(ModelRun):
    periods_dict:dict[str, dict[str, list[str]]]
    nc_file_folder:str
    period:str

    def __init__(self, periods_dict:dict[str, dict[str, list[str]]], nc_file_folder:str, cut_to_period:bool):
        super().__init__('DKHYPE', 'Q[mmday]', 'Q_sim[mmday]')
        self.periods_dict = periods_dict
        self.nc_file_folder = nc_file_folder
        self.cut_to_period = cut_to_period

        self.min_obs_in_basin = 10
        self.all_basins = list(self.periods_dict.keys())
        self.label = 'DKHYPE'

    def load_results(self):
        for basin in self.all_basins:
            file_name = pj(self.nc_file_folder, basin + '.nc')
            if os.path.exists(file_name):
                dataset = xr.open_dataset(file_name)
                ds = dataset.load()
                dataset.close()

                if self.cut_to_period:
                    basin_times = self.periods_dict[basin]
                    start_date = basin_times['start_dates'][0]
                    end_date = basin_times['end_dates'][0]
                    ds = ds.sel(date=slice(start_date, end_date))
                
                ds = ds.dropna(dim='date', how='any')
            
                if len(ds['date']) <= self.min_obs_in_basin:
                    self.no_observations_basins.append(basin)
                    continue

                ds[self.qsim_label] = ds[self.qsim_label]# * 1e6
                self.nc_file_dataset[basin] = ds
            else:
                print(f"Basin {basin} file does not exist")
                self.no_observations_basins.append(basin)
                self.all_basins.remove(basin)
                continue

        self.filtered_basins = list(self.nc_file_dataset.keys())
            


class NeuralHydrology_DKSUBSET_ModelRun(ModelRun):
    periods_dict:dict[str, dict[str, list[str]]]
    results_file:str
    period:str

    def __init__(self, model_run_name:str, periods_dict:dict[str, dict[str, list[str]]], results_file:str, cut_to_period:bool):
        super().__init__(model_run_name, 'Q[mmday]_obs', 'Q[mmday]_sim')
        self.periods_dict = periods_dict
        self.results_file = results_file
        self.cut_to_period = cut_to_period

        self.min_obs_in_basin = 10
        
        self.all_basins = list(self.periods_dict.keys())

    
    def load_results(self):
        with open(self.results_file, "rb") as fp:
            self.nc_file_dataset = pickle.load(fp)

        for basin in self.all_basins:
            if basin not in self.nc_file_dataset:
                self.no_observations_basins.append(basin)
                continue

            if self.cut_to_period:
                basin_times = self.periods_dict[basin]
                start_date = basin_times['start_dates'][0]
                end_date = basin_times['end_dates'][0]
                self.nc_file_dataset[basin]['1D']['xr'] = self.nc_file_dataset[basin]['1D']['xr'].sel(date=slice(start_date, end_date))
            
            self.nc_file_dataset[basin] = self.nc_file_dataset[basin]['1D']['xr'].dropna(dim='date', how='any')
            
            if len(self.nc_file_dataset[basin]['date']) <= self.min_obs_in_basin:
                self.no_observations_basins.append(basin)
                del self.nc_file_dataset[basin]
                
        self.filtered_basins = list(self.nc_file_dataset.keys())


class NeuralHydrology_USCAMELS_ModelRun(ModelRun):
    results_file:str
    period:str

    def __init__(self, model_run_name:str, results_file:str, cut_to_period:bool):
        super().__init__(model_run_name, 'QObs(mm/d)_obs', 'QObs(mm/d)_sim')
        self.results_file = results_file
        self.cut_to_period = cut_to_period

        self.min_obs_in_basin = 10

    """
    If period is None, then the whole dataset is used. Otherwise, the period is used to filter the dataset.
    """
    def load_results(self):
        with open(self.results_file, "rb") as fp:
            self.nc_file_dataset = pickle.load(fp)
        
        self.all_basins = list(self.nc_file_dataset.keys())
        for basin in self.all_basins:
            if self.cut_to_period:
                start_date = pd.Timestamp(datetime.strptime('01/10/1989', '%d/%m/%Y'))
                end_date = pd.Timestamp(datetime.strptime('30/09/1999', '%d/%m/%Y'))
                self.nc_file_dataset[basin]['1D']['xr'] = self.nc_file_dataset[basin]['1D']['xr'].sel(date=slice(start_date, end_date))
            self.nc_file_dataset[basin] = self.nc_file_dataset[basin]['1D']['xr'].dropna(dim='date', how='any')
            
        if len(self.nc_file_dataset[basin]['date']) <= self.min_obs_in_basin:
            self.no_observations_basins.append(basin)
            del self.nc_file_dataset[basin]

        self.filtered_basins = list(self.nc_file_dataset.keys())