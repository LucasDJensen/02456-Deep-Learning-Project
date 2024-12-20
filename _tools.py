import netCDF4 as nc
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from os.path import join as pj
import re

REPO_FOLDER = r'C:\Users\lucas\Documents\02456-Deep-Learning-Project'
DATA_PROJECT_PATH = REPO_FOLDER
BASIN_ATTRIBUTE_FILE = pj(REPO_FOLDER, r'DK_SUBSET_CAMELS\attributes\basin_attributes.csv')
NC_FILES_FOLDER = pj(REPO_FOLDER, r'DK_SUBSET_CAMELS\time_series')

THIS_PROJECT_PATH = REPO_FOLDER
PROJECT_FIGURES_PATH = pj(THIS_PROJECT_PATH, "figures")
PROJECT_GENERATED_PATH = pj(THIS_PROJECT_PATH, "generated")
PROJECT_DATA_CONFIG_PATH = pj(THIS_PROJECT_PATH, 'data_config')

TRAIN_BASIN_FILE = pj(PROJECT_DATA_CONFIG_PATH, 'train_basin_file.txt')
TEST_BASIN_FILE = pj(PROJECT_DATA_CONFIG_PATH, 'test_basin_file.txt')
VALIDATION_BASIN_FILE = pj(PROJECT_DATA_CONFIG_PATH, 'validation_basin_file.txt')

STATION_MAPPING_FILE = pj(PROJECT_GENERATED_PATH, "station_mapping_table.csv")
PER_BASIN_DATE_RANGES_FILE = pj(PROJECT_GENERATED_PATH, 'per_basin_date_ranges.csv')
PER_BASIN_QSIM_DATE_RANGES_FILE = pj(PROJECT_GENERATED_PATH, 'per_basin_qsim_date_ranges_from.csv')


# Make directories if they don't exist
for path in [PROJECT_FIGURES_PATH, PROJECT_GENERATED_PATH, PROJECT_DATA_CONFIG_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)


class Tools:
    def __init__(self):
        if not os.path.isfile(STATION_MAPPING_FILE):
            print('Station mapping file does not exist. Gathering and saving...')
            self.generate_station_mapping_file()
            print('Station mapping file created.')

        self.station_id_table = pd.read_csv(STATION_MAPPING_FILE, delimiter=',')
        self.all_catchment_ids = self.station_id_table['obsstednr'].values

        gauged_catchments_df = pd.read_csv(BASIN_ATTRIBUTE_FILE, delimiter=',')
        gauged_catchments_df.rename(columns={"Unnamed: 0":"catchment_id"}, inplace=True) 
        new_column = []
        for catchment in gauged_catchments_df.iterrows():
            new_column.append(self.station_id_table[self.station_id_table["obsstednr"] == catchment[1]["catchment_id"]]["Id15_v30"].values[0])
        gauged_catchments_df["Id15_v30"] = new_column
        self.gauged_catchments_df = gauged_catchments_df
        self.gauged_catchment_ids = gauged_catchments_df['catchment_id'].values
        self.gauged_catchment_Id15_v30 = gauged_catchments_df['Id15_v30'].values

    
    def generate_station_mapping_file(self):
        ### ENSURE THESE ARE CORRECT IF YOU WANT TO CREATE STATION MAPPING TABLE
        Id15v3_to_hypeid_csv = '/dmidata/projects/hydrologi/LSTM_highres/gis_data/station_mapping/Id15v3_to_hypeid.csv'
        map_ove_station_to_dkhype_newids_all_csv = '/dmidata/projects/hydrologi/LSTM_highres/gis_data/station_mapping/map_ove_station_to_dkhype_newids_all.csv'

        id15v3_to_hypeid = pd.read_csv(Id15v3_to_hypeid_csv, delimiter=',', dtype={'Id15_v30': np.int64, 'hypeid': np.int64})
        hypeid_to_dkhypeid = pd.read_csv(map_ove_station_to_dkhype_newids_all_csv, delimiter=',', dtype={'hypeid': np.int64, 'obsstednr': np.int64})
        all_catchment_id_mapping = pd.merge(id15v3_to_hypeid, hypeid_to_dkhypeid, on='hypeid', how='left') # For some reason this has a float value...
        all_catchment_id_mapping.to_csv(STATION_MAPPING_FILE, index=False)


    def load_date_range_dataframe(self,manipulate_period=False):
        if manipulate_period:
            file = PER_BASIN_QSIM_DATE_RANGES_FILE
        else:
            file = PER_BASIN_DATE_RANGES_FILE

        if not os.path.exists(file):
            tools.generate_date_range(file, manipulate_period=manipulate_period)

        df = pd.read_csv(file, parse_dates=['start_date', 'end_date'])
        print(f"Date ranges loaded from {file}")
        
        return df

    def generate_date_range(self, file, manipulate_period=False):
        exclude_list = []
        # exclude_list = [
        #     # No training data ->
        #     '21000807', '25000592', '26000111', '26000170', '36000015', '36000016', '49000092', '52000064', '53000029'
        #     # No validation data ->
        #     '18000471', '21000526', '21000647', '21000861', '23000173', '41000140', '41000141', '42000124', '42000125', '45000041', '46000174', '52000064'
        # ]


        # Initialize a list to store date range information
        date_ranges = []

        # Loop through all .nc files in the directory
        for filename in sorted(os.listdir(NC_FILES_FOLDER)):
            if filename.endswith('.nc'):
                basin = filename.split('.')[0]
                if basin in exclude_list:
                    # print(f"Skipping {filename}")
                    continue
                file_path = os.path.join(NC_FILES_FOLDER, filename)
                try:
                    # Open the netCDF file
                    nc_file = nc.Dataset(file_path, 'r')

                    # Extract the 'date' variable
                    date_var = nc_file.variables['date']

                    # Get the 'units' attribute
                    units_attr = date_var.units

                    # Extract the reference date from the units string
                    match = re.match(r'days since (\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)', units_attr)
                    if match:
                        ref_date_str = match.group(1)
                        ref_date = datetime.strptime(ref_date_str, '%Y-%m-%d %H:%M:%S' if ' ' in ref_date_str else '%Y-%m-%d')
                    else:
                        print(f"Could not parse date units in file {filename}")
                        continue
                    
                    # Get the number of observations
                    num_observations = date_var.shape[0]
                    start_date = ref_date
                    end_date = ref_date + timedelta(days=int(num_observations - 1))

                    if manipulate_period:
                        # Setting fixed start date
                        if start_date.year <= 2010:
                            print(f"Discarding preceding 2010 in the dataset: {filename}")
                            # Set start date to 2002-01-01
                            start_date = datetime(year = 2011, month=1, day=1)
                        
                        if start_date > end_date:
                            print(f"Discarding the first year in the dataset makes start_date go beyond end_date: {filename}")
                            continue

                    # Append to the list
                    date_ranges.append({
                        'filename': filename,
                        'start_date': start_date,
                        'end_date': end_date
                    })

                    # Close the netCDF file
                    nc_file.close()
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

        # Convert the list to a DataFrame
        df = pd.DataFrame(date_ranges)
        df['date_diff'] = (df['end_date'] - df['start_date']).dt.days

        # Sort the DataFrame by start_date
        df.sort_values(by='filename', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save the DataFrame to a CSV file
        df.to_csv(file, index=False)
        print(f"Date ranges saved to {file}")


def netcdf_to_dataframe(netcdf_file):
    # Open the NetCDF file
    dataset = nc.Dataset(netcdf_file)
    
    # Extract variables
    data_dict = {}
    for var_name in dataset.variables:
        var_data = dataset.variables[var_name][:]
        
        # Flatten the data if it has more than one dimension
        if var_data.ndim > 1:
            var_data = var_data.flatten()
        
        data_dict[var_name] = var_data
    
    # Create the DataFrame
    min_length = min(len(v) for v in data_dict.values())  # Ensure all arrays are of the same length
    df = pd.DataFrame({k: v[:min_length] for k, v in data_dict.items()})
    
    return df

tools = Tools()

if __name__ == "__main__":
    tools.load_date_range_dataframe(manipulate_period=True)
    # print(tools.all_catchment_ids)
    # print(tools.gauged_catchment_ids)
    # print(tools.gauged_catchment_Id15_v30)