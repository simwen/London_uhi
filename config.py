#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:08:27 2024

@author: Sim
"""

from datetime import date
import os

# Input USGS EarthExplorer credentials https://earthexplorer.usgs.gov
username = "SimWen"
password = "..."

# Set directories
BASE_DIR = '/Users/Sim/Documents/Other/Programming/Personal Projects/Climate & Health/Landsat'
DATA_DIR = os.path.join(BASE_DIR,'data')

today = date.today()
RUN_NAME = f"{today}-1"

RAW_DIR = os.path.join(DATA_DIR, RUN_NAME,'raw')
INT_DIR = os.path.join(DATA_DIR, RUN_NAME, 'intermediate')
FIN_DIR = os.path.join(DATA_DIR, RUN_NAME, 'final')

