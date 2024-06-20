#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:08:27 2024

@author: Sim
"""
print("1: Starting config.py")

from datetime import date
import os

# Input USGS EarthExplorer credentials https://earthexplorer.usgs.gov
username = "SimWen"
password = "..."

# Set directories
BASE_DIR = 'Z:\Resources\Personal\Simeon Wentzel\london_uhi_data'
DATA_DIR = os.path.join(BASE_DIR,'data')

today = date.today()
RUN_NAME = f"{today}-1"

ALL_RUNS = os.path.join(DATA_DIR, 'all_runs')
RAW_DIR = os.path.join(DATA_DIR, RUN_NAME,'raw')
INT_DIR = os.path.join(DATA_DIR, RUN_NAME, 'intermediate')
FIN_DIR = os.path.join(DATA_DIR, RUN_NAME, 'final')

directories = [DATA_DIR, ALL_RUNS, RAW_DIR, INT_DIR, FIN_DIR]

# Create each directory if it doesn't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

print("1: Finished config.py")

import extract
import analyse

extract.run(username = username, 
           password = password, 
           today = today, 
           ALL_RUNS = ALL_RUNS,
           RAW_DIR = RAW_DIR, 
           INT_DIR = INT_DIR)

analyse.run(today = today, 
            ALL_RUNS = ALL_RUNS,
            INT_DIR = INT_DIR,
            FIN_DIR = FIN_DIR)

