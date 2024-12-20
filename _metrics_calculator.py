
"""
Helper functions for evaluating various types of performance scores across simulations.

The main block script reads all the simulated time series from a HYPE simulation and a Qobs file.
The user of the script then defines time periods (start and end timestamps) for the calibration and validation periods,
and then the script calculates performance metrics such as KGE and NSE for calibration and validation periods.

PERFORMANCE METRICS SHOULD HAVE f(sim,obs,*args) AS INPUTS SO THEY CAN BE USED SYSTEMATICALLY IN SEVERAL SCRIPTS

###############                      FOR CALIBRATION               #####################################################
THIS SCRIPT HAS TO BE LOCATED IN THE HYPE MODEL FOLDER TO WORK IN THE CALIBRATION FRAMEWORK (MAKE A COPY)
The main folder path - NEEDS TO BE os.path.abspath(os.path.dirname(__file__)) if run in OSTRICH

This script reads all the simulated time series from a HYPE simulation and a Qobs file.
The user of the script then defines time periods (start and end timestamps) for the calibration and validation periods,
and then the script calculates performance metrics such as KGE and NSE for calibration and validation periods.

The function used in the code are also located in hype_output_tools, but need to be defined here so that they can be run
independently within the calibration framework

WARNING: Rounding the score in the export can make the ostrich calibration insensitive to small changes
"""

import numpy as np
import pandas as pd
import os

def drop_missing_obs_values(sim, obs):
    """Drop rows where obs has nan values"""
    nans_in_obs = np.isnan(obs)
    obs = obs[~nans_in_obs]
    sim = sim[~nans_in_obs]
    return sim, obs


def drop_missing_obs_or_sim_values(sim, obs):
    """Drop rows where obs has nan values"""
    nans_in_obs = np.isnan(obs) | np.isnan(sim)
    obs = obs[~nans_in_obs]
    sim = sim[~nans_in_obs]
    return sim, obs

def get_index_of_percentile(sorted_percentiles, target_percentile, tolerance=0.1):
    """find the index for the closest percentile,
    Tolerance is the allowable difference between target_percentile and the closest match found in input percentiles.
    Percentiles should be specified as fractional 0-1 values
    tolerance=0.1 allows up to 10% difference between found and target,
    e.g. if target_percentiles=0.1 and tolerance=0.1 looks for a percentile between 0.09 and 0.11"""

    idx = np.abs(sorted_percentiles - target_percentile).argmin()
    if np.abs(sorted_percentiles[idx] / target_percentile - 1)<tolerance:
        return idx
    else:
        return None

def get_threshold_from_percentile(obs, percentile, tolerance=0.1):
    """ gets a threshold flow from a percentile value """
    obs_sorted, obs_exceedance = get_fdc(obs)
    percentile_idx = get_index_of_percentile(obs_exceedance, target_percentile=percentile, tolerance=tolerance)
    if percentile_idx is not None:
        return obs_sorted[percentile_idx]
    else:
        return None

def get_relative_error(sim_value, obs_value, percent=True):
    """ does the relative error of an indicator - by default as percent (%)"""
    factor = 100 if percent==True else 1
    return (sim_value-obs_value)/obs_value*factor

def get_peaks(arr):
    """A peak discharge is the discharge at a time step of which both the previous and the following time step have a
    lower discharge (from https://hess.copernicus.org/articles/17/1893/2013/hess-17-1893-2013.pdf)
    returns only peak values from an array"""
    peaks = []
    arr = np.array(arr)
    # Loop through the array from the second element to the second last element
    if len(arr)>2:
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks.append(arr[i])
    return pd.Series(peaks)

def kge_calc(sim, obs):
    """ Calculate KGE.  Assumes that sim does not have nan values. Drop rows in both obs and sim where obs is nan."""

    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)

    r = np.ma.corrcoef(np.ma.masked_invalid(sim),
                       np.ma.masked_invalid(obs))[0][1]

    kge_corr = (r - 1) ** 2
    kge_var = (np.var(sim) ** 0.5 / np.var(obs) ** 0.5 - 1) ** 2
    kge_mean = (np.mean(sim) / np.mean(obs) - 1) ** 2

    kge = 1 - np.sqrt(kge_corr + kge_var + kge_mean)
    return kge


def nse_calc(sim, obs):
    """Calculates NSE. Assumes that sim does not have nan values. Drop rows in both obs and sim where obs is nan."""

    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)

    errors = sim - obs
    sum_squared_errors = np.sum(errors ** 2)

    obs_mean = np.mean(obs)
    obs_around_mean = obs - obs_mean
    var_obs = np.sum(obs_around_mean ** 2)

    nse = 1 - sum_squared_errors / var_obs
    return float(nse)


def mse_calc(sim, obs):
    """Calculates MSE. Assumes that sim does not have nan values. Drop rows in both obs and sim where obs is nan."""

    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)

    errors = sim - obs
    sum_squared_errors = np.sum(errors ** 2)
    mse = sum_squared_errors / len(obs)
    return mse


def fbal_calc(sim, obs):
    """Calculates Fbal, the long-term waterbalance of surface runoff. 
        Assumes that sim does not have nan values. Drop rows in both obs and sim where obs is nan."""

    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)
    
    fbal = 100*(np.mean(sim)-np.mean(obs))/np.mean(obs)
    return fbal


def get_fdc(q, use_log=False):
    """ returns sorted values and exceedance likelihood for Flow duration curve (FDC) for non na values in q"""
    q_sorted = np.sort(q[~np.isnan(q)])
    q_ranked = np.array(range(1, q_sorted.size + 1))
    exceedance_sorted = 1 - q_ranked / q_ranked.size
    if use_log:  # log transform
        q_sorted = np.log10(q_sorted)
        fdc = np.log10(exceedance_sorted)
    return q_sorted, exceedance_sorted


def get_cdf(data_in: pd.Series, as_percentage=False):
    """Get cdf for non na values in one column data"""

    data_in = np.array(data_in.astype(float).dropna())
    xvals = np.sort(data_in)
    yvals = np.arange(1, len(xvals) + 1) / float(len(xvals))
    if as_percentage:
        yvals = yvals * 100
    return xvals, yvals


def peak_max_percent_error(sim, obs):
    """peak maximum percent error (%) (Qmax,model - Qmax,meas)/Qmax,meas * 100"""
    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)
    # return metric
    return (max(sim)-max(obs))/max(obs)*100

def peak_distribution(sim, obs):
    """ This signature shows whether the peak discharges are of equal height; therefore, only the peak discharges are taken
    into account. A peak discharge is the discharge at a time step of which both the previous and the following time step have a
    lower discharge. From these peak discharges a flow duration curve is constructed and the average slope between the 10th
    and 50th percentile is taken as the measure for this signature.
    https://hess.copernicus.org/articles/17/1893/2013/hess-17-1893-2013.pdf
    Rem: this script does not account for if series have missing values - will introduce mini-bias
    """
    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)

    # Get slope of peaks
    def get_slope_between_peaks(obs):
        # Get peak flows
        obs_peaks = get_peaks(obs)

        #get fdc
        obs_peak_sorted, obs_peak_exceedance = get_fdc(obs_peaks, use_log=False)

        # find id of 10th and 50th percentiles
        obs_ten_idx = get_index_of_percentile(obs_peak_exceedance, 0.1, tolerance=0.15)
        obs_fifty_idx = get_index_of_percentile(obs_peak_exceedance, 0.5, tolerance=0.15)

        # Return slope
        if obs_ten_idx is not None and obs_fifty_idx is not None:
            return (obs_peak_sorted[obs_ten_idx] - obs_peak_sorted[obs_fifty_idx]) / (0.9 - 0.5)
        else:
            return np.nan

    # Calculate slopes
    obs_slope = get_slope_between_peaks(obs)
    sim_slope = get_slope_between_peaks(sim)

    # Return slope difference (%)
    return (sim_slope-obs_slope)/obs_slope*100

def nse_of_fdc(sim, obs):
    """For this signature a flow duration curve is constructed from all the discharge data. The Nash–Sutcliffe efficiency
    (Nash and Sutcliffe, 1970) between the observed and modelled flow duration curve is taken as the evaluation criterion.
    Flow duration curves are frequently used hydrological signatures to evaluate the overall behaviour of a catchment.
    https://hess.copernicus.org/articles/17/1893/2013/hess-17-1893-2013.pdf (p7)
    """
    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)

    # Get flow duration curves ( which is basically sorted values)
    sim_sorted = np.sort(sim)
    obs_sorted = np.sort(obs)

    # Get NSE of FDC
    return nse_calc(sim_sorted, obs_sorted)

def rising_limb_density(sim, obs):
    """
    Like the autocorrelation, this signature is an indication of the smoothness of the hydrograph, but the RLD is averaged over
    the total period and is completely independent of the flow volume (Shamir et al., 2005). This signature is calculated
    by dividing the number of peaks by the total time the hydrograph is rising. Therefore, the RLD is the inverse of the
    mean time to peak. Together with RLD also DLD (declining limb density) has been used before for supporting the
    calibration process
    https://hess.copernicus.org/articles/17/1893/2013/hess-17-1893-2013.pdf (p7)
    Rem: this script does not account for if series have missing values - will introduce mini-bias
    """
    def get_rld(obs):
        arr = np.array(obs)
        # Get peaks
        obs_peaks = get_peaks(arr)
        # Count peaks
        count_peaks = len(obs_peaks)
        # Get rising time
        obs_rising = [arr[k]<arr[k+1] for k in range(len(arr)-1)]
        obs_rising_count = sum(obs_rising)
        # Return rising limb density : number of peaks / rising time  (in day-1) - inverse of average rising time
        return count_peaks/obs_rising_count

    # Drop rows where obs has nan values
    sim, obs = drop_missing_obs_values(sim, obs)

    # Calculate rising lamb density for obs and sim
    obs_rld = get_rld(obs)
    sim_rld = get_rld(sim)

    return (sim_rld-obs_rld)/obs_rld*100

def get_count_hit_miss_events(sim, obs, percentile=0.01, threshold_value=0, tolerance=0.1):
    """Calculate the hit rate over a specified threshold or use_percentile
    hit is defined as (sim & obs)>threshold"""
    # Get threshold
    if threshold_value==0:
        threshold_value = get_threshold_from_percentile(obs, percentile, tolerance=tolerance)
        if threshold_value is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Identify events over the threshold in observations
    obs_over_threshold = obs > threshold_value
    sim_over_threshold = sim > threshold_value

    # Total number of events exceeding threshold in observations
    n_obs_events = np.sum(obs_over_threshold)

    # Calculate total model events (model above threshold)
    n_sim_events = np.sum(sim_over_threshold)

    # Calculate hits (both sim and obs exceed threshold)
    n_hits = np.sum(sim_over_threshold & obs_over_threshold)

    # Calculate false alarms (sim exceeds threshold, obs does not)
    n_false = np.sum(sim_over_threshold & ~obs_over_threshold)

    # Calculate misses (obs exceeds threshold, sim does not)
    n_misses = np.sum(~sim_over_threshold & obs_over_threshold)

    # Chance forecast (n_obs_events*n_sim_events/sample_size)
    n_chance = n_obs_events*n_sim_events/len(obs)

    return n_obs_events, n_sim_events, n_hits, n_false, n_misses, n_chance

def hit_rate_over_threshold(sim, obs, percentile=0.01, threshold_value=0):
    """
    Calculate the hit rate over a specified threshold or use_percentile
    hit is defined as (sim & obs)>threshold
    """
    # Get stats
    n_obs_events, n_sim_events, n_hits, n_false, n_misses, n_chance = get_count_hit_miss_events(sim, obs,
                                                                                                percentile=percentile,
                                                                                                threshold_value=threshold_value)

    # Calculate hit rate
    hit_rate = n_hits / n_obs_events if n_obs_events > 0 else np.nan
    return hit_rate

def success_ratio_over_threshold(sim, obs, percentile=0.01, threshold_value=0):
    """
    Calculate the success ratio over a specified threshold or percentile
    """
    # Get stats
    n_obs_events, n_sim_events, n_hits, n_false, n_misses, n_chance = get_count_hit_miss_events(sim, obs,
                                                                                                percentile=percentile,
                                                                                                threshold_value=threshold_value)

    # Return success ratio
    success_ratio = n_hits / n_sim_events if n_sim_events>0 else np.nan
    return success_ratio

def false_alarm_over_threshold(sim, obs, percentile=0.01, threshold_value=0):
    """
    Calculate the false alarm rate over a specified threshold or percentile
    """
    # Get stats
    n_obs_events, n_sim_events, n_hits, n_false, n_misses, n_chance = get_count_hit_miss_events(sim, obs,
                                                                                                percentile=percentile,
                                                                                                threshold_value=threshold_value)

    # Calculate false alarm rate
    false_alarm_rate = n_false / n_obs_events if n_obs_events > 0 else np.nan
    return false_alarm_rate

def critical_success_index_over_threshold(sim, obs, percentile=0.01, threshold_value=0, tolerance=0.1):
    """
    Calculate the critical success index over a specified threshold or percentile
    CSI = (hits) / (hits + false alarms + misses)
    """
    # Get stats
    n_obs_events, n_sim_events, n_hits, n_false, n_misses, n_chance = get_count_hit_miss_events(sim, obs,
                                                                                                percentile=percentile,
                                                                                                threshold_value=threshold_value, tolerance=tolerance)
    hit_false_miss = n_hits + n_false + n_misses
    # Calculate critical success index
    critical_success_index = n_hits / hit_false_miss if hit_false_miss > 0 else np.nan
    return critical_success_index, n_hits, n_false, n_misses


def equitable_threat_score_over_threshold(sim, obs, percentile=0.01, threshold_value=0):
    """
    Calculate the equitable threat score over a specified threshold or percentile
    https://resources.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos2/uos2_ko4.htm
    ETS = (hits - hits expected by chance) / (hits + false alarms + misses – hits expected by chance)
    ar = (total forecasts of the event) * (total observations of the event) / (sample size)
    The number of forecasts of the event correct by chance, a r , is determined by assuming that the forecasts are
    totally independent of the observations, and forecast will match the observation only by chance.
    This is one form of an unskilled forecast, which can be generated by just guessing what will happen.
    The ETS has a range of -1/3 to 1, but the minimum value depends on the verification sample climatology.
    For rare events, the minimum ETS value is near 0, while the absolute minimum is obtained if the event has a
    climatological frequency of 0.5, and there are no hits. If the score goes below 0 then the chance forecast is
    preferred to the actual forecast, and the forecast is said to be unskilled.
    """
    # Get stats
    n_obs_events, n_sim_events, n_hits, n_false, n_misses, n_chance = get_count_hit_miss_events(sim, obs,
                                                                                                percentile=percentile,
                                                                                                threshold_value=threshold_value)

    # Calculate equitable_threat_score
    hit_false_miss_chance = n_hits + n_false + n_misses - n_chance
    equitable_threat_score = (n_hits-n_chance) / hit_false_miss_chance if hit_false_miss_chance > 0 else np.nan
    return equitable_threat_score


def pierce_skill_score(sim, obs, percentile=0.01, threshold_value=0):
    """
    Calculate the Pierce Skill Score (PSS) over a specified threshold.
    """
    # Calculate hit rate and false alarm rate
    hit_rate = hit_rate_over_threshold(sim, obs, percentile=percentile, threshold_value=threshold_value)
    false_alarm_rate = false_alarm_over_threshold(sim, obs, percentile=percentile, threshold_value=threshold_value)

    # Calculate PSS
    pss = hit_rate - false_alarm_rate
    return pss

def symetric_extremal_dependence_index(sim, obs, percentile=0.01, threshold_value=0):
    """
    Calculate the Symmetric Extremal Dependence Index (SEDI) over a specified threshold.
    https://nhess.copernicus.org/articles/24/1415/2024/
    """

    # Calculate hit rate and false alarm rate
    hr = hit_rate_over_threshold(sim, obs, percentile=percentile, threshold_value=threshold_value)
    far = false_alarm_over_threshold(sim, obs, percentile=percentile, threshold_value=threshold_value)

    # Calculate SEDI
    if hr <= 0 or hr >= 1 or far <= 0 or far >= 1:
        sedi = np.nan
    else:
        sedi = (np.log(far) - np.log(hr) - np.log(1 - far) + np.log(1 - hr)) / (
                    np.log(far) + np.log(hr) + np.log(1 - far) + np.log(1 - hr))

    return sedi

def calc_bfi_one_series(flow_data, window_size=5):
    """
    Calculation of the base flow index for one timeseries based on the UKIH method (UK Institute of Hydrology, 1980). Originally
    implemented in the research repository. The method is described in Piggott et al. (2005) https://doi.org/10.1623/hysj.2005.50.5.911
    Parameters
    ----------
    flow_data
        A pd.Series of flow data. Probably, a list would work as well.
    window_size: int
        Size of a moving window used in BFI calculations. Must be equal to or larger than 5. The UKIH method explicitly defines
        segments to be 5-day windows. Other baseflow calculation methods define segments as a function og drainage area.
    Returns
    -------
    base_flow_index: float
        The base flow index for the flow_data input series ranging from 0-1
    """

    # window_size must be >=5
    if window_size < 5:
        raise ValueError(f'Argument "window_size" in function bfi must be equal to or larger than 5. Got {window_size}')
    flow_data = flow_data + 5  # Model has problems with zeros, hence we lift the graph
    flow_data = flow_data.dropna()  # Removing all NaN values before first data point
    if len(flow_data) == 0: # return nan if all values where nan
        return np.nan
    # The first and last couple of data points cannot be used, so pad the graph to get more data.
    q_padded = np.append(flow_data[29::-1], flow_data)
    q_padded = np.append(q_padded, flow_data[-30:])
    number_of_points = q_padded.shape[0]

    # Calculate the base flow  #########################
    min_q_in_window = -np.ones((round(number_of_points / window_size), 1))
    min_q_idx = -np.ones((round(number_of_points / window_size), 1))
    window_no = 0
    turning_points = np.zeros(0)
    turning_points_index = np.zeros(0)

    # Find the largest number we can iterate to and still have a whole window (so that last window doesn't go
    # out bounds)
    max_iteration_no = int(np.floor(number_of_points / window_size) * window_size)
    # Iterate over all the windows and test if we have turning points. If we have, save them.
    for i in range(0, max_iteration_no, window_size):
        min_q_in_window[window_no] = min(q_padded[i:i + window_size])  # Find minimum flow value in window
        min_q_idx[window_no] = np.argmin(q_padded[i:i + window_size])  # Find index of minimum value in window

        # Test the turning point condition: 0.9*y_i < min(y_(i-1), y(i+1)) from UKIH method (note that index window
        # number (i) in the array is shifted one place the in condition below).
        if window_no > 1:
            if (min_q_in_window[window_no - 1] * 0.9 < min_q_in_window[window_no - 2] and
                    min_q_in_window[window_no - 1] * 0.9 < min_q_in_window[window_no]):
                # If we have turning point, add it and its index to the lists
                turning_points = np.append(turning_points, min_q_in_window[window_no - 1])
                turning_points_index = np.append(turning_points_index, i - window_size + min_q_idx[window_no - 1])
        window_no += 1

    # Calculate time index from first to last turning point, so we have a point for each day in the period
    t_ind = [i for i in range(int(turning_points_index[0]), int(turning_points_index[-1]))]
    t_ind = np.append(t_ind, t_ind[-1] + 1)

    # Calculate the base flow as straight lines between the turning points, one point per day (t_ind)
    temp_base_flow = np.interp(t_ind, turning_points_index, turning_points)
    temp_flow = q_padded[t_ind]  # get the real data for the same about of time as temp_base_flow
    # In cases where interpolated values are higher than real ones, use real ones
    temp_base_flow[temp_base_flow > temp_flow] = temp_flow[temp_base_flow > temp_flow]

    base_flow = temp_base_flow[int(30 - turning_points_index[0]):]  # remove some of the padding at the start
    base_flow = base_flow[:flow_data.size]  # cut so the new flow array is not longer than original

    if base_flow.size == flow_data.size:
        pass
    else:
        flow_data = flow_data[1:]

    flow_data = flow_data - 5  # Lower graphs again
    base_flow = base_flow - 5

    # Some formating of arrays
    nan_array_end = np.ones(len(flow_data) - len(base_flow))
    nan_array_end[:] = np.nan

    base_flow = np.append(base_flow, nan_array_end)

    base_flow[np.isnan(flow_data)] = np.nan
    flow_data[np.isnan(base_flow)] = np.nan

    # Calculate Base Flow Index, BFI  ###################################
    base_flow_index = np.nansum(base_flow) / np.nansum(flow_data)

    return base_flow_index


def append_to_csv(file_path, df):
    """append results to csv if file does not exist, otherwise, creates a new"""
    if not os.path.isfile(file_path):
        df.to_csv(file_path, mode='w', header=True)
    else:
        df.to_csv(file_path, mode='a', header=False)


def get_next_iteration(file_path):
    if not os.path.isfile(file_path):
        return 1
    existing_data = pd.read_csv(file_path)
    return existing_data.shape[0] + 1


if __name__ == "__main__":
    # main folder - NEEDS TO BE os.path.abspath(os.path.dirname(__file__)) if run in OSTRICH
    # otherwise can just put a path to model run e.g. #"/dmidata/projects/hydrologi/DKHYPE/Calibration/DKHYPE"
    main_folder = os.path.abspath(os.path.dirname(__file__))
    # Result and observed data file directory/paths
    results_path = os.path.join(main_folder, 'results', 'timeCOUT.txt')
    q_obs_path = os.path.join(main_folder, "Qobs.txt")
    # number of digits in export files _
    # WARNING, a value below 4 might impact calibration since model might become insensitive to small changes
    round_to = 5

    # file containing the map (or link) between dkhype catchments and observations (e.g. in Qobs.txt)
    map_hypeid_obsid_path = os.path.join(main_folder, "map_ove_station_to_dkhype_newids.csv")

    #  define observation station names (link to dkhype ids is made through map)
    stations = ['2000005', '3000003', '6000001', '7000003', '18000077', '21000461',
                '22000225', '22001541', '22000062', '25000082', '31000027', '36000008',
                '38000024', '26000082', '27000671', '28000001', '66000014', '52000020',
                '48000007', '58000047', '59000006', '61000012', '64000025', '47000036',
                '47000037', '46000017', '45000003', '45000043', '46000030', '42000021',
                '41000391', '37000038', '38000020', '34000023', '34000003', '32000001',
                '32000004']

    # Define times you want to calculate scores for
    # list of time horizons to evaluate {name: date_range}
    time_ranges = {'cal': pd.date_range(start='2015-01-01', end='2020-12-31', freq='D'),
                   'val': pd.date_range(start='2021-01-01', end='2022-12-31', freq='D'),
                   }

    # export path for scores
    score_total_path = os.path.join(main_folder, "scores_total.csv")
    score_total_it_path = os.path.join(main_folder, "scores_total_it.csv")
    score_all_path = os.path.join(main_folder, "scores_all.csv")

    # Performance metrics
    percentile = 0.01  # used for hit/miss/ over threshold metrics to define a threshold for peaks
    performance_metrics = {'KGE': kge_calc,
                           'NSE': nse_calc,
                           'fbal': fbal_calc,
                           'nseoffdc': nse_of_fdc,
                           'BFI': lambda sim, obs: calc_bfi_one_series(sim) - calc_bfi_one_series(obs),
                           'peakerror': peak_max_percent_error,
                           'peakdistribution': peak_distribution,
                           'RLD': rising_limb_density,
                           'HR': lambda sim, obs: hit_rate_over_threshold(sim, obs, percentile=percentile),
                           'SR': lambda sim, obs: success_ratio_over_threshold(sim, obs, percentile=percentile),
                           'FA': lambda sim, obs: false_alarm_over_threshold(sim, obs, percentile=percentile),
                           'CSI': lambda sim, obs: critical_success_index_over_threshold(sim, obs, percentile=percentile),
                           'ETS': lambda sim, obs: equitable_threat_score_over_threshold(sim, obs, percentile=percentile),
                           'PSS': lambda sim, obs: pierce_skill_score(sim, obs, percentile=percentile),
                           'SEDI': lambda sim, obs: symetric_extremal_dependence_index(sim, obs, percentile=percentile)
                           }

    # load map and set observation id as index
    if map_hypeid_obsid_path is None:  # return dummy mapping keeping ids
        map_hypeid_obsid = {s: s for s in stations}
    else:
        map_hypeid_obsid = pd.read_csv(map_hypeid_obsid_path).astype(str)
        map_hypeid_obsid = map_hypeid_obsid.set_index('obsstednr').to_dict()['catchment']

    # Load data
    # load hype data
    cout = pd.read_csv(results_path, sep="\t", header=1, skiprows=0)
    cout["DATE"] = pd.to_datetime(cout["DATE"], format="%Y-%m-%d")

    # load qobs data
    q_obs_raw = pd.read_csv(q_obs_path, header=0, sep="\t")
    q_obs_raw["date"] = pd.to_datetime(q_obs_raw["date"], format="%Y-%m-%d")
    q_obs_raw[q_obs_raw == -9999] = np.nan

    # Loop across performance metrics and time ranges
    score_dict = {}
    for perf_name in performance_metrics.keys():
        for time_name in time_ranges.keys():
            cout_time = cout[cout['DATE'].isin(time_ranges[time_name])]
            q_obs_time = q_obs_raw[q_obs_raw['date'].isin(time_ranges[time_name])]
            score_dict.update({
                perf_name + '_' + time_name: [
                performance_metrics[perf_name](cout_time[map_hypeid_obsid[s]], q_obs_time[s]) for s in stations]
            })

    # define a dataframe with performance metrics for both calibration and validation periods
    scores = pd.DataFrame(score_dict)

    # Get the next iteration number
    iteration_number = get_next_iteration(score_total_it_path)

    # format scores dataframe
    scores = scores.round(round_to)
    scores.index = stations
    scores.index.name = "Station"
    scores["Iteration"] = iteration_number
    scores = scores.reset_index()
    scores = scores.set_index(["Iteration", "Station"])
    append_to_csv(score_all_path, scores)

    # latest total score for calibration framework (read by objective function)
    # calculate average score
    scores_total = scores.mean()
    scores_total.round(round_to).to_csv(score_total_path)

    # total score including previous iterations of calibration framework (for visualization after)
    scores_total_it = scores.dropna().mean().to_frame().transpose()
    scores_total_it.index = [iteration_number]
    scores_total_it.index.name = "Iteration"
    scores_total_it = scores_total_it.round(round_to)
    append_to_csv(score_total_it_path, scores_total_it)