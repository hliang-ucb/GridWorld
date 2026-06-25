import numpy as np
import pynwb
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from dateutil import tz
from my_utils import nwb
import h5py
from mat73 import loadmat
from uuid import uuid4
from tqdm import tqdm
from my_utils import pyMonkeyLogic as pyML
from my_utils.graph import grid_graph
from my_utils.utils import movmean, find_nearest
from my_utils import utils
import networkx as nx
from my_utils.saccades import SaccadeKmeansGrid2
from pathlib import Path
from my_utils.lfp import find_ripple_events



def load_eventcodes(data_dir):
    """_summary_

    Args:
        data_dir (_type_): Path object to the data directory, to a .npy file 

    Returns:
        _type_: Load in eventcodes from directory
    """
    eventTable = np.load(data_dir / 'sync_event_codes.npy')
    return eventTable


def load_bhv_file(data_dir):
    """_summary_

    Args:
        data_dir (_type_):  Path object to the data directory, to a .h5 file 

    Returns:
        _type_: Load in behavioral data from directory
    """
    bhv_filename = utils.get_files_by_type(data_dir, ".h5")[0]
    bhv_file = h5py.File(bhv_filename, "r")
    
    return bhv_file


def load_kilosort_data(results_dir, region="", max_time=None):
    """
    Load Kilosort data from the specified directory.

    Parameters:
        results_dir (str): The directory path where the Kilosort results are stored.
        region (str, optional): The region name associated with the data. Defaults to an empty string.
        max_time (int, optional): The maximum time (in milliseconds) to include in the spike table. 
            If not provided, the time of the last spike plus a 10ms pad will be used.

    Returns:
        spike_table (numpy.ndarray): The spike table containing spike information for each cluster. Shape is n_clusters x max_time (ms)
        unitNames (pandas.DataFrame): The metadata for each cluster, including firing rate, amplitude, 
            channel locations, cluster groups, and region.
    """
    
    # Make Spike Table
    cluster_ids = np.load(results_dir / 'spike_clusters.npy')
    ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
    spike_times = np.load(results_dir / 'spike_times.npy')

    n_clusters = len(np.unique(cluster_ids))
    arange_clusters = np.arange(n_clusters)
    cluster_remapping = np.ones((max(cluster_ids) + 1), dtype=int) * -100
    cluster_remapping[np.unique(cluster_ids)] = arange_clusters

    isi_violations = []
    for i in np.unique(cluster_ids):
        if i == -1:
            continue
        isi_violations.append(((np.diff(spike_times[cluster_ids == i])/ops["fs"]*1000) < 1).sum()/(cluster_ids == i).sum())
    isi_violations = np.array(isi_violations, dtype=float)
    # convert spike times to ms
    spike_times = np.array(spike_times/ops["fs"]*1000, dtype=int)
    
    if max_time is None:
        # if no end time is given, find the time of the last spike, and add a 10ms pad to the end 
        max_time = np.ceil(np.max(spike_times)).astype(int) + 5000
    else: 
        cluster_ids = cluster_ids[spike_times < max_time]
        spike_times = spike_times[spike_times < max_time]
    # initialize spike table
    spike_table = np.zeros([n_clusters, max_time], dtype=bool)
    # fill in spike table
    spike_table[cluster_remapping[cluster_ids], spike_times] = 1

    # get unitnames metadata 
    unitNames_base = pd.read_csv(results_dir / 'cluster_info.tsv', sep="\t") # cluster contamination percentages
    unitNames = unitNames_base.copy()
    #unitNames.drift[unitNames.drift == "False"] = 0
    unitNames.drift = unitNames.drift.fillna(0).astype(float)

#    for column in unitNames.columns:
#        if (unitNames[column].dtype == 'object'):
#            unitNames[column] = unitNames[column].astype("string")

    
    columns_to_drop = ["dift", "droft", "to_split", "sh", "group", "KSLabel"]
    for col in columns_to_drop:
        if col in unitNames.columns:
            unitNames = unitNames.drop(col, axis=1)
    
    unitNames["region"] = region
    unitNames["group"] = unitNames_base["group"].astype(str)
    unitNames["KSLabel"] = unitNames_base["KSLabel"].astype(str)
    unitNames["isi_violations"] = isi_violations
    for column in unitNames.columns:
        print(column, unitNames[column].dtype)
    
    
    """
    camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values # cluster amplitudes
    templates =  np.load(results_dir / 'templates.npy') # cluster templates and channel locations
    contamPct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep="\t") # cluster contamination percentages
    cluster_groups = pd.read_csv(results_dir / "cluster_group.tsv", sep="\t") # cluster groups
    firing_rates = np.unique(cluster_ids, return_counts=True)[1] * 1000 / spike_times.max() # cluster firing rates
    
    # make unitNames dataframe
    unitNames = contamPct.copy()
    unitNames["firing_rate"] = firing_rates
    unitNames["amplitude"] = camps 
    unitNames = unitNames.merge(cluster_groups, left_on="cluster_id", right_on="cluster_id")
    unitNames["region"] = region
    # get channel locations for each cluster
    #ch_start = np.zeros([n_clusters, ], dtype=int)
    #ch_stop = np.zeros([n_clusters, ], dtype=int)
    #for i in range(n_clusters):
    #    ch_start[i] = int(np.where(templates[i][0, :])[0][0])
    #    ch_stop[i] = int(np.where(templates[i][0, :])[0][-1])
    #unitNames["ch_start"] = ch_start
    #unitNames["ch_stop"] = ch_stop
    #unitNames["ch"] = np.array((ch_start + ch_stop) // 2, dtype=int)

    n_units = templates.shape[0]
    best_channels = np.zeros([n_units, ], dtype=int)
    best_wvs = np.zeros([n_units, templates.shape[1]], dtype=float)
    for unit in range(n_units):
        wv = templates[unit, ...] # get waveforms
        troughs = np.min(wv, axis=0)
        peaks = np.max(wv, axis=0)
        diff = peaks - troughs
        best_channels[unit] = np.argmax(diff)
        best_wvs[unit] = wv[:, best_channels[unit]]
    unitNames["ch"] = best_channels
    unitNames["waveform"] = best_wvs.tolist()
    
    """
    return spike_table, unitNames


def load_lfp_data(roi_dir):
    """
    Load LFP data from the specified ROI directory.

    Parameters:
        roi_dir (str): The path to the ROI directory.

    Returns:
        tuple: A tuple containing the LFP data and the corresponding timestamps.
    """
    lfp_data = np.load(roi_dir / "sync_lfp.npy")
    lfp_ts = np.load(roi_dir / "sync_lfp_ts.npy")
    return lfp_data, lfp_ts


def refactor_bhv_data(bhv_file, npx_event_codes):
    
    n_trials = int(bhv_file["ML"]["TrialRecord"]["CurrentTrialNumber"][:][0][0])
    
    print("refactoring eventcodes + eye data")
    #processed_eye_data = np.ones([plx_file["spikeTable"].shape[1], 2])*-10000
    processed_eye_data = []
    processed_eye_timestamps = []
    npx_eventcodes = npx_event_codes[:, 0]
    npx_eventtimes = npx_event_codes[:, 1]
    npx_trial_start = npx_eventtimes[np.where(npx_eventcodes == 9)]
    npx_trial_end = npx_eventtimes[np.where(npx_eventcodes == 18)]
    
    n_npx_trials = len(npx_trial_start)
    
    if n_npx_trials != n_trials:
        raise ValueError("Number of trials in behavioral file and npx file do not match")
    
    ml_codes_all = []
    ml_ts_all = []
    for _trial in tqdm(range(n_trials), unit="trial(s)"):
        
        ml_codes = pyML.get_bhvcodes(bhv_file, _trial)
        npx_start = int(npx_trial_start[_trial])
        npx_end = int(npx_trial_end[_trial])
        _, eye = nwb.get_ml_trial_info(bhv_file, _trial)        
        
        ml_codes_all.append(ml_codes["numbers"].reshape(-1, 1))
        ml_ts = ml_codes["times"]
        ml_ts = ml_ts - ml_ts[0] + npx_start
        ml_ts_all.append(ml_ts.reshape(-1, 1))
        
        n_eye_samples = eye["data"].shape[0]
        eye_timestamps = np.linspace(npx_start, npx_end, n_eye_samples).reshape(-1, 1)
        processed_eye_timestamps.append(eye_timestamps)
        processed_eye_data.append(eye["data"])
   
    ml_codes_all, ml_ts_all = np.ravel(np.vstack(ml_codes_all)), np.ravel(np.vstack(ml_ts_all))
    eye_data, eye_ts = np.vstack(processed_eye_data), np.ravel(np.vstack(processed_eye_timestamps))
    return ml_codes_all, ml_ts_all, eye_data, eye_ts


def add_intervals(nwbfile, bhv_file):
    """
    Add trial intervals and custom event time intervals to the NWBFile object.

    Parameters:
        nwbfile (NWBFile): The NWBFile object to which the intervals will be added.

    Returns:
        nwbfile (NWBFile): The updated NWBFile object.
    """
    # Get set of acquired timeseries
    n_trials = pyML.get_n_trials(bhv_file)
    uservar_names = pyML.get_uservars(bhv_file, 0).keys()
    timeseries_set = []
    for key in nwbfile.acquisition.keys():
        timeseries_set.append(nwbfile.acquisition[key])

    # Get eventcodes and times
    ev_data = nwbfile.acquisition["events"].data
    ev_ts = nwbfile.acquisition["events"].timestamps
    
    # Get trial start and end times
    bhv_trial_start = ev_ts[np.where(ev_data == 9)]
    bhv_trial_end = ev_ts[np.where(ev_data == 18)]
    
    # Create trial intervals
    nwbfile.add_trial_column(name="trial", description="trial number")
    nwbfile.add_trial_column(name="trialerror", description="trialerror")
    for _name in uservar_names:
        nwbfile.add_trial_column(name=_name, description=_name)
    nwbfile.add_trial_column(name="nodes", description="list of nodes encountered in the trial")

    print("Creating Trial Events:")
    # iterate through trials and add trial data
    for _trial in tqdm(range(n_trials), unit="trial(s)"):
        ev_start, ev_end = bhv_trial_start[_trial], bhv_trial_end[_trial]
        bhv, _ = nwb.get_ml_trial_info(bhv_file, _trial)
        
        nwbfile.add_trial(start_time=float(ev_start),
                        stop_time=float(ev_end)+50,
                        trial=_trial,
                        timeseries=timeseries_set,
                        id=_trial,
                        **bhv)
    
    # Create custom timeintervals around important trial events.
    node_on = nwb.NodeInterval(name="node_on", description="TimeInterval parsing the onset of each node stimulus. Indexes a 500ms time window around the stimulus", uservar_names=uservar_names)
    action_on = nwb.NodeInterval(name="action_on", description="TimeInterval parsing the onset of each active fixation. Indexes a 500ms time window around the action onset", uservar_names=uservar_names)
    reward_on = nwb.NodeInterval(name="reward_on", description="TimeInterval parsing the onset of each reward epoch. Indexes a 500ms time window around the reward cue onset", uservar_names=uservar_names)

    print("Creating Custom Event TimeIntervals")
    # create action, node, and reward timeintervals
    fixation_df = pd.DataFrame()
    for trial in tqdm(nwbfile.trials.to_dataframe().query("trialerror < 2").trial, unit="trial(s)"):
        # Get eventcodes + eventtimes from the trial 
        evs, ets = nwbfile.trials["timeseries"][trial][1].data, nwbfile.trials["timeseries"][trial][1].timestamps
        
        # find the node onset times
        first_node_code, last_node_code = 31, 49
        node_idx = np.where((evs <= last_node_code) & (evs >= first_node_code))[0]
        node_onset_times = ets[node_idx]
        nodes = np.array(evs[node_idx] - first_node_code, dtype=int)
        
        jackpot = pyML.get_jackpot_steps(bhv_file, trial)
        
        # because of how the fixations are coded, need to separate the first step onset and the active steps 
        first_step_onset_time = ets[np.where((evs == 82))[0][0]] - 300
        nav_onset_times = ets[np.where(evs == 92)[0]] - 300
        active_onset_times = np.hstack([first_step_onset_time, nav_onset_times])
        
        # Find the reward onset times
        reward_cue_onset_time = ets[np.where(evs == 70)[0]][0]

        # Initialize Graph
        G_tele = grid_graph(4, 4, tele=[0, 15])
        G_spatial = grid_graph(4, 4, tele=None)
        n_steps = len(node_onset_times)
        for _step in range(n_steps):
            bhv, _ = nwb.get_ml_trial_info(bhv_file, trial)
            
            current_node = nodes[_step]
            bhv["jackpot"] = jackpot[_step]
            graph_distance = nx.shortest_path_length(G=G_tele, source=current_node, target=bhv["target"])
            spatial_distance = nx.shortest_path_length(G=G_spatial, source=current_node, target=bhv["target"])
            t_start, t_stop = node_onset_times[_step]-500, node_onset_times[_step]+500
            node_on.add_interval(start_time=t_start,
                            stop_time=t_stop,
                            t_on=node_onset_times[_step],
                            window_size=500,
                            trial=trial,
                            step=_step,
                            graph_distance=graph_distance,
                            spatial_distance=spatial_distance,
                            node=current_node,
                            timeseries=timeseries_set,
                            **bhv)

            t_start, t_stop = active_onset_times[_step]-500, active_onset_times[_step]+500        
            action_on.add_interval(start_time=t_start,
                            stop_time=t_stop,
                            t_on=active_onset_times[_step],
                            window_size=500,
                            trial=trial,
                            step=_step,
                            graph_distance=graph_distance,
                            spatial_distance=spatial_distance,
                            node=current_node,
                            timeseries=timeseries_set,
                            **bhv)
            
        t_start, t_stop = reward_cue_onset_time - 500, reward_cue_onset_time + 500
        reward_on.add_interval(start_time=t_start,
                            stop_time=t_stop,
                            t_on=reward_cue_onset_time,
                            window_size=500,
                            trial=trial,
                            step=_step,
                            graph_distance=graph_distance,
                            spatial_distance=spatial_distance,
                            node=current_node,
                            timeseries=timeseries_set,
                            **bhv)
        
        trial_df = nwbfile.trials[trial]
        eyes, eye_ts = trial_df["timeseries"][trial][2].data, trial_df["timeseries"][trial][2].timestamps   
        model = SaccadeKmeansGrid2(step_length=5.4, filter_order=2, min_saccade_distance=1.25, min_saccade_duration=10, build_table=True, use_session_clock=True)
        model.fit(eyes[:, 0].copy(), eyes[:, 1].copy(), ev_codes=evs, ev_times=ets)
        model.fixations["trial"] = trial
        model.fixations = model.fixations.rename({"node":"fix_node"}, axis=1)
        trial_df = trial_df.rename({"start_time":"trial_start_time", "stop_time":"trial_stop_time"}, axis=1).drop("timeseries", axis=1)
        fixation_df = pd.concat([fixation_df, pd.merge(left=trial_df, right=model.fixations, on="trial")])


    fixation_df["window_size"] = 500
    fixations = pynwb.epoch.TimeIntervals(name="fixations", description="Fixations identified by utils.Saccade.SaccadeKmeansGrid2")
    
    for column_name in fixation_df.columns:
        fixations.add_column(name=column_name, description=column_name)
    
    n_fixations = fixation_df.shape[0]        
    for _fix in range(n_fixations):
        row = fixation_df.iloc[_fix, :]
        
        start_time = row.t_on + 0.0
        stop_time = row.t_off + 0.0
        fixations.add_interval(start_time=start_time, stop_time=stop_time, timeseries = timeseries_set, **row.to_dict())
        
    
    nwbfile.add_time_intervals([node_on, action_on, reward_on, fixations])    
    return nwbfile 


def create_nwb(data_dir, write=False, output_file=None, regions=["OFC", "HPC"]):

    ### Load in all the data ###
    
    # load behavioral data
    eventTable = load_eventcodes(data_dir)
    bhv_file = load_bhv_file(data_dir)
    ev_data, ev_ts, eye_data, eye_ts = refactor_bhv_data(bhv_file=bhv_file, npx_event_codes=eventTable)
    print("preprocessed eventcodes + eye data")

    roi_directories = utils.get_subdirectories(data_dir)

    lfp_data, spike_table, unitNames, lfp_ts = [], [], [], []
    for roi_dir in roi_directories:
        if roi_dir.__str__().__contains__("imec0"):
            roi_ = regions[0]
        elif roi_dir.__str__().__contains__("imec1"):
            roi_ = regions[1]
            
        roi_dir = Path(roi_dir)
        results_dir = Path(roi_dir / 'kilosort4')
        roi_spike_table, roi_unitNames = load_kilosort_data(results_dir, region=roi_)
        roi_lfp_data, roi_lfp_ts = load_lfp_data(roi_dir)
        
        lfp_data.append(roi_lfp_data)
        spike_table.append(roi_spike_table)
        unitNames.append(roi_unitNames)
        lfp_ts.append(roi_lfp_ts)
        print(roi_spike_table.shape)
        
    # check that lfp timestamps match up 
    if len(lfp_data) > 1:
        if ~np.all(lfp_ts[0] == lfp_ts[1]):
            raise ValueError("LFP timestamps do not match between regions")
        
        if spike_table[0].shape[1] != spike_table[1].shape[1]:
            max_time = np.min([spike_table[0].shape[1], spike_table[1].shape[1]])
            longer_dataset = np.argmax([spike_table[0].shape[1], spike_table[1].shape[1]])
            spike_table[longer_dataset] = spike_table[longer_dataset][:, :max_time]
            print("Reshaped Spike Data")
            
        if lfp_data[0].shape[1] != lfp_data[1].shape[1]:
            max_time = np.min([lfp_data[0].shape[1], lfp_data[1].shape[1]])
            longer_dataset = np.argmax([lfp_data[0].shape[1], lfp_data[1].shape[1]])
            lfp_data[longer_dataset] = lfp_data[longer_dataset][:, :max_time]
            print("Reshaped LFP Data")
            
        
    
    # refactor data structures
    lfp_data, spike_table, unitNames, lfp_ts = np.vstack(lfp_data), np.vstack(spike_table), pd.concat(unitNames), lfp_ts[0]
    
    # find ripples
    ripple_ts, ripple_df = find_ripple_events(lfpTable=lfp_data, offset=int(lfp_ts[0]))
    max_ch = np.argmax(ripple_df.groupby("channel").count().iloc[:, 0])
    ripple_df = ripple_df.loc[ripple_df.channel == max_ch, :]
    
    # Create Timeseries
    neural_time_series = pynwb.base.TimeSeries(name="neural", description="raw neural spiking data", data=spike_table.T, unit="ms", rate=1.0)
    eventcode_time_series = pynwb.base.TimeSeries(name="events", description="eventcodes", data=ev_data, timestamps=ev_ts, unit="ms")
    eyetracking_time_series = pynwb.base.TimeSeries(name="eyes", description="eyetracking data", data=eye_data, timestamps=eye_ts, unit="ms")
    lfp_time_series = pynwb.base.TimeSeries(name="lfp", description="Realigned LFP data, 0.5-500Hz", data=lfp_data.T, timestamps=lfp_ts, unit="ms")
    ripple_time_series = pynwb.base.TimeSeries(name="ripples", description="Identified SWR timestamps", data=ripple_ts, rate=1.0, unit="ms" )
    
    timeseries_set = [neural_time_series, eventcode_time_series, eyetracking_time_series, lfp_time_series, ripple_time_series]
    print("Created Timeseries")
    
    
    #Initialize NWB File 
    date_string = str(data_dir).split("_")[-2]
    m, d, y = int(date_string[0:2]), int(date_string[2:4]), int(date_string[4:6])
    session_date = datetime.datetime(y+2000, m, d, tzinfo=tz.gettz("US/Pacific"))
    session_description = "Bart, Dual NP recording OFC/HPC"
    
    if data_dir.__str__().__contains__("v3"):
        session_id = "TeleWorld_4x4_v3"
    else:
        session_id = "TeleWorld_4x4"

    nwbfile = pynwb.NWBFile(session_description=session_description, identifier=str(uuid4()), session_start_time=session_date, session_id=session_id,
                            lab="Wallis Lab", institution="Univeristy of California, Berkeley")

    for ts in timeseries_set:
        nwbfile.add_acquisition(ts)
    print("Created NWB File")
    

    nwbfile = add_intervals(nwbfile, bhv_file)
    print("Added custom time intervals")
    
    # Add unitnames to nwbfile
    n_units = unitNames.shape[0]
    for column_name in list(unitNames.columns):
        nwbfile.add_unit_column(column_name, description=column_name)
    for unit in range(n_units):
        row = unitNames.iloc[unit, :]
        nwbfile.add_unit(id=unit, **row.to_dict())
    print("Added unitNames")
    
    # Add ripple intervals to nwbfile
    ripple_interval = pynwb.epoch.TimeIntervals(name="ripple",
                                        description="intervals for each detected swr")
    for val in ripple_df.columns.values:
        ripple_interval.add_column(name=val, description=val)
        
    for _row in tqdm(range(ripple_df.shape[0])):
        row_dict=  ripple_df.iloc[_row, :].to_dict()
        window_padding = 200
        ripple_interval.add_interval(start_time=float(row_dict["ripple_on"] - window_padding),
                            stop_time=float(row_dict["ripple_off"] + window_padding),
                            timeseries=timeseries_set,
                            **row_dict)
    nwbfile.add_time_intervals(ripple_interval)
    print("Added SWR interval objects")
    
    del lfp_data, spike_table, eye_data, ripple_ts, ripple_df, lfp_ts, ev_data
    print("Deleted large data structures")
    
    if write: 
        with pynwb.NWBHDF5IO(output_file, "w") as io:
            io.write(nwbfile)
        print("File Saved!")
    return nwbfile 


def create_nwb_only_spikes(data_dir, write=False, output_file=None, regions=["OFC", "HPC"], probes_to_process=[0, 1]):

    ### Load in all the data ###
    
    # load behavioral data
    eventTable = load_eventcodes(data_dir)
    bhv_file = load_bhv_file(data_dir)
    ev_data, ev_ts, eye_data, eye_ts = refactor_bhv_data(bhv_file=bhv_file, npx_event_codes=eventTable)
    print("preprocessed eventcodes + eye data")

    roi_directories = utils.get_subdirectories(data_dir)
    roi_directories = sorted(roi_directories)
    roi_directories = [roi_directories[i] for i in probes_to_process]
    print(roi_directories)
    spike_table, unitNames = [], []
    for roi_dir in roi_directories:
        if roi_dir.__str__().__contains__("imec0"):
            roi_ = regions[0]
        elif roi_dir.__str__().__contains__("imec1"):
            roi_ = regions[1]
            
        roi_dir = Path(roi_dir)
        results_dir = Path(roi_dir / 'kilosort4')
        roi_spike_table, roi_unitNames = load_kilosort_data(results_dir, region=roi_)

        spike_table.append(roi_spike_table)
        unitNames.append(roi_unitNames)
        print(roi_spike_table.shape)
        
    # check that lfp timestamps match up 
    if len(spike_table) > 1:
        if spike_table[0].shape[1] != spike_table[1].shape[1]:
            max_time = np.min([spike_table[0].shape[1], spike_table[1].shape[1]])
            longer_dataset = np.argmax([spike_table[0].shape[1], spike_table[1].shape[1]])
            spike_table[longer_dataset] = spike_table[longer_dataset][:, :max_time]
            print("Reshaped Spike Data")
        
    
    # refactor data structures
    spike_table, unitNames = np.vstack(spike_table), pd.concat(unitNames)
    # find ripples

    # Create Timeseries
    neural_time_series = pynwb.base.TimeSeries(name="neural", description="raw neural spiking data", data=spike_table.T, unit="ms", rate=1.0)
    eventcode_time_series = pynwb.base.TimeSeries(name="events", description="eventcodes", data=ev_data, timestamps=ev_ts, unit="ms")
    eyetracking_time_series = pynwb.base.TimeSeries(name="eyes", description="eyetracking data", data=eye_data, timestamps=eye_ts, unit="ms")
 
    timeseries_set = [neural_time_series, eventcode_time_series, eyetracking_time_series]
    print("Created Timeseries")
    
    
    #Initialize NWB File 
    date_string = str(data_dir).split("_")[4]
    m, d, y = int(date_string[0:2]), int(date_string[2:4]), int(date_string[4:6])
    session_date = datetime.datetime(y+2000, m, d, tzinfo=tz.gettz("US/Pacific"))
    session_description = "Bart, Dual NP recording OFC/HPC"
    
    if data_dir.__str__().__contains__("v3"):
        session_id = "TeleWorld_4x4_v3"
    else:
        session_id = "TeleWorld_4x4"

    nwbfile = pynwb.NWBFile(session_description=session_description, identifier=str(uuid4()), session_start_time=session_date, session_id=session_id,
                            lab="Wallis Lab", institution="Univeristy of California, Berkeley")

    for ts in timeseries_set:
        nwbfile.add_acquisition(ts)
    print("Created NWB File")
    

    nwbfile = add_intervals(nwbfile, bhv_file)
    print("Added custom time intervals")
    
    # Add unitnames to nwbfile

    n_units = unitNames.shape[0]
    for column_name in list(unitNames.columns):
        nwbfile.add_unit_column(column_name, description=column_name)
    for unit in range(n_units):
        row = unitNames.iloc[unit, :]
        nwbfile.add_unit(id=unit, **row.to_dict())
    print("Added unitNames")

    # Add ripple intervals to nwbfile
    del spike_table, eye_data, ev_data
    print("Deleted large data structures")
    
    if write: 
        with pynwb.NWBHDF5IO(output_file, "w") as io:
            io.write(nwbfile)
        print("File Saved!")
    return nwbfile 
    
    
if __name__ == "__main__":
    """
    session_paths = ["/media/eric/EH SSD/London/London_TeleWorld_4x4_100124_g0"]
    save_names = ["/home/eric/VSCode-Neurons/data/processed/London_TeleWorld_4x4_100124_spikes.nwb"]
    """
    """
    session_paths = ["/media/eric/EH SSD/London/London_TeleWorld_4x4_092124_g0",
                     "/media/eric/EH SSD/London/London_TeleWorld_4x4_092324_g0",
                     "/media/eric/EH SSD/London/London_TeleWorld_4x4_092524_g0",]
    
    
    
    save_names = ["/home/eric/VSCode-Neurons/data/processed/London_TeleWorld_4x4_092124_spikes.nwb",
                  "/home/eric/VSCode-Neurons/data/processed/London_TeleWorld_4x4_092324_spikes.nwb",
                  "/home/eric/VSCode-Neurons/data/processed/London_TeleWorld_4x4_092524_spikes.nwb",]
    """
    #session_paths = ["/home/eric/London_Sorted_Data/London_TeleWorld_4x4_101124_g0"]
    #save_names = ["/home/eric/VSCode-Neurons/data/london/neural/London_TeleWorld_4x4_101124_spikes.nwb",]
    """
    session_paths = ["/media/eric/London Data/London/London_TeleWorld_4x4_100324_g1",
                     "/media/eric/London Data/London/London_TeleWorld_4x4_100924_g0",
                     "/media/eric/London Data/London_TeleWorld_4x4_101124_g0"]
    """
    session_paths = [#"/home/eric/Bart_Sorted_Data/Bart_TeleWorld_4x4_NPX2_043024_g0",
                     #"/home/eric/Bart_Sorted_Data/Bart_TeleWorld_4x4_NPX2_050824_g0",
                     "/home/eric/Bart_Sorted_Data/Bart_TeleWorld_4x4_NPX2_051024_g0"]
    session_paths = ['/media/eric/partition_1/ACC Data/London/London_TeleWorld_4x4_102424_ACC_HPC_g1']
    save_names = [#"/home/eric/VSCode-Neurons/data/bart_I/npx/Bart_TeleWorld_4x4_043024_spikes.nwb",
                  #"/home/eric/VSCode-Neurons/data/bart_I/npx/Bart_TeleWorld_4x4_050824_spikes.nwb",
                  "/home/eric/VSCode-Neurons/data/bart_I/npx/Bart_TeleWorld_4x4_051024_spikes.nwb"]
    save_names = ["/media/eric/partition_1/ACC Data/London/London_TeleWorld_4x4_102424_ACC_HPC_g1/London_TeleWorld_4x4_102424_ACC_spikes.nwb"]
    
    for data_path, save_path in zip(session_paths, save_names):    
        data_dir = Path(data_path)
        create_nwb_only_spikes(data_dir, write=True, output_file=save_path, regions=["ACC", "HPC"], probes_to_process=[0])

    
    

    

