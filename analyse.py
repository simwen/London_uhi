#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:19:43 2024

@author: Sim
"""

from scipy.stats import percentileofscore
import rasterio as rio
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point

today = '2024-04-30'

## 1. Create geodataframe -----------------------------------------------

# Load the raster data & transformation info
with rio.open(os.path.join(INT_DIR,f'{today}_celcius.tif')) as src_avg:
    
    raster_data = src_avg.read(1)  
    transform = src_avg.transform

# Get the width and height of the raster
width = raster_data.shape[1]
height = raster_data.shape[0]

# Get the coordinates of the pixel centers
x_coords, y_coords = zip(*[transform * (j + 0.5, i + 0.5) for i in range(height) for j in range(width)])

# Flatten the raster values
pixel_values = raster_data.flatten()

# Create pandas dataframe
london_temps = pd.DataFrame({'Pixel_Value': pixel_values, 'longitude': x_coords, 'latitude': y_coords})

# Convert to a GeoDataFrame of temperatures and points (pixel coords)
geometry_london_temps = [Point(xy) for xy in zip(london_temps['longitude'], london_temps['latitude'])]
gdf_london_temps = gpd.GeoDataFrame(london_temps, geometry=gpd.GeoSeries(geometry_london_temps))

if not os.path.exists(os.path.join(INT_DIR, f'shapefile {today}')):
    os.makedirs(os.path.join(INT_DIR, f'shapefile {today}'))
gdf_london_temps.to_file(os.path.join(INT_DIR, f'shapefile {today}', f"celcius_gdf.shp"))

                         
## 2. Average by LAD -----------------------------------------------

def average_by_geog(geography = 'LAD'):
    # collapse the GeoDataFrame based on 'LAD22NM'
    london_geog = london_lsoas_bng
    if geography == 'LAD':
        london_geog = london_geog.dissolve(by='LAD22NM')
    london_geog = london_geog.reset_index()
    
    # empty list for LAD temperatures
    avg_temp_list = []
    
    # For each LAD, creates boolean of which pixel in LAD, then averages temperatures for each LAD
    for index, polygon in london_geog.iterrows():
        # creates boolean mask for if points in gdf_london_temps are within the each london_lsoas_bng polygon
        mask = gdf_london_temps.within(polygon['geometry'])
        # calculates average pixel value (temp) where mask is true 
        avg_temp = gdf_london_temps.loc[mask, 'Pixel_Value'].mean()
        avg_temp_list.append(avg_temp)
    
    # Add the average temp values to df1
    london_geog.loc[:, 'avg_temp'] = avg_temp_list
    
    # Clean and save out
    london_geog = london_geog.drop(['level_0', 'index'], axis=1)
    
    return(london_geog)
    
# Save averaged data
output_file_path = os.path.join(FIN_DIR,f'{today}_{geography}_temps')
london_geog.to_file(output_file_path, driver='ESRI Shapefile')
    