#!/usr/bin/env python
# coding: utf-8

# load a session


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import glob
import os
import pynwb
import graph
import h5py
from tqdm import tqdm
from pathlib import Path 
from datetime import datetime
import platform

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import OneHotEncoder

def which_system():
    if platform.system()=='Darwin':
        dir = '/Volumes/Extreme SSD'
    else:
        dir = 'D:'

    return dir

def initiate_directory(animal, date):  # region, epoch, query

    dir = which_system()

    # wrapper to easily fetch data from one session
    
    # combine all sessions
    if animal=='London':
        
        DATA_DIR = Path(f'{dir}/Teleworld/london/neural')

    if animal=='Bart':

        if datetime.strptime(date, "%m%d%y")<datetime.strptime('050824', "%m%d%y"):
            DATA_DIR = Path(f'{dir}/Teleworld/bart_I/processed')
        else:
            DATA_DIR = Path(f'{dir}/Teleworld/bart_I/npx')

    return DATA_DIR


def load_nwbfile(animal, date):
    DATA_DIR = initiate_directory(animal, date)
    os.chdir(DATA_DIR)
    
    filename = glob.glob('*%s*' % date) # sorted()
    nwbfile = pynwb.NWBHDF5IO(filename[0], "r").read()

    return nwbfile



def load_beh_neural(animal, date, region, epoch, query):

    # wrapper to easily fetch data from one session
    
    DATA_DIR = initiate_directory(animal, date)
    os.chdir(DATA_DIR)
    
    filename = glob.glob('*%s*' % date) # sorted()
    nwbfile = pynwb.NWBHDF5IO(filename[0], "r").read()
    
    spikes, df, unitNames = get_spike_table(nwbfile, region, epoch, query=query)
    df.insert(0,'Session',date)
    unitNames.insert(0,'Session',date)
    unitNames.insert(0,'Animal',animal)

    trial_cc = pd.DataFrame(
    df.drop_duplicates('trial')
      .groupby('block')
      .cumcount()
      .rename('trial_in_block')
)

    trial_cc['trial'] = df.trial.unique()

    df = df.merge(trial_cc, on='trial')
    
    return {'spikes':spikes, 
            'beh':df, 
            'unitNames':unitNames}



def load_LFP(date):

    # get LFP data
    lfpFile = h5py.File(f'{dir}/Teleworld/bart_I/raw/spikes/Bart_TeleWorld_v13_%s-spikes.mat' % date, 'r')
    timestamps = np.array(lfpFile['/lfpTimeStamps']).ravel()
    lfpData = lfpFile['/lfpTable']
    
    return timestamps, lfpData


#def load_spikes(date):


def get_spike_table(nwbfile, region, epoch, query="", window_size=0, unit_params = {"drift": 2, "min_fr": 1}):
    
    """_summary_
    Returns:
        (np.array, pd.DataFrame) : spike table aligned to the specified epoch, dataframe with step metadata
        
    """
    
    assert epoch in nwbfile.intervals.keys(), f"Epoch {epoch} not found in nwbfile intervals. Available epochs: {nwbfile.intervals.keys()}"
    
    assert region in ["HPC", "OFC", "both"], f"Region {region} not found. Available regions: ['HPC', 'OFC']"

    
    unitNames = nwbfile.units.to_dataframe()
    
    # get all good units
    if region == 'both':
        unit_idx =  np.where((unitNames.group == "good") & (unitNames.drift <= unit_params["drift"]) & (unitNames.fr >= unit_params["min_fr"]))[0]
    else:
        unit_idx = np.where((unitNames.region == region) & (unitNames.group == "good") & (unitNames.drift <= unit_params["drift"]) & (unitNames.fr >= unit_params["min_fr"]))[0]
    unitNames = unitNames.iloc[unit_idx, :].reset_index(drop=True)

    if ~np.isin('unitNumber',unitNames.columns):
        unitNames['unitNumber']=unitNames['cluster_id']    

    
    # subsample an epoch with a specific query
    if query == "":
        df = nwbfile.intervals[epoch].to_dataframe()#["timeseries"]
    else:
        df = nwbfile.intervals[epoch].to_dataframe().query(query)#["timeseries"]

    neural_timeseries_index = 0
    sample_index = df.index
    
    if epoch!='fixations':        
        epoch_win_size = df["window_size"].values[0]
        spikes = np.zeros((len(sample_index), epoch_win_size*2, len(unit_idx)), dtype=np.float32)
        print("Building Spike Table")
        if window_size != 0:
            for i, _sample in tqdm(enumerate(sample_index)):
                sample_data = df["timeseries"][_sample][neural_timeseries_index].data[:, unit_idx]
                sample_data = movmean(sample_data.T, window_size).T
                spikes[i, ...] = sample_data.reshape(1, epoch_win_size*2, -1)
        else:
            for i, _sample in tqdm(enumerate(sample_index)):
                sample_data = df["timeseries"][_sample][neural_timeseries_index].data[:, unit_idx]
                spikes[i, ...] = sample_data.reshape(1, epoch_win_size*2, -1)
    

    if epoch=='fixations':
        
        spikes = np.zeros((len(df), len(unit_idx)), dtype=np.float32)
        
        for i, _sample in tqdm(enumerate(sample_index)):
            sample_data = df["timeseries"][_sample][neural_timeseries_index].data[:, unit_idx]
            spikes[i,:] = sample_data.mean(axis=0)
        
        df['planning'] = (df.duration<300) & (df.active_prob<0.2)
        df['choice'] = (df.duration>300) & (df.active_prob>0.2)
    
    df = graph.append_use_tele(df)
    df = df.drop(columns=["timeseries"])
    
    return spikes, df, unitNames



# def load_theta(date):

# def load_SWR(date):