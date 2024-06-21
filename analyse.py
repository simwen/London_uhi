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
from tqdm import tqdm
from shapely import wkt

def run(today, ALL_RUNS, INT_DIR, FIN_DIR):

    print("##################################################")
    print("3: Starting analyse.py")
    print("##################################################")

    ## 1. Create geodataframe -----------------------------------------------

    print('Creating geodataframe')

    # Load the raster data & transformation info
    with rio.open(os.path.join(INT_DIR,f'{today}_avg_celcius.tif')) as src_avg:
        
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
    gdf_london_temps.to_file(os.path.join(INT_DIR, f'shapefile {today}', "celcius_gdf.shp"))

                            
    ## 2. Average by LAD/LSOA -----------------------------------------------

    print('Average by LAD or LSOA')
    london_lsoas_lads_bng =  pd.read_csv(os.path.join(ALL_RUNS, 'london_lsoas_lads_bng.csv'))
    london_lsoas_lads_bng['geometry'] = london_lsoas_lads_bng['geometry'].apply(wkt.loads)

    def average_by_geog(dataframe = london_lsoas_lads_bng, geography = 'LAD'):
        
        dataframe = gpd.GeoDataFrame(dataframe, geometry='geometry')

        # collapse the GeoDataFrame based on 'LAD22NM'
        if geography == 'LAD':
            dataframe = dataframe.dissolve(by='LAD22NM')
        dataframe = dataframe.reset_index()
        
        # empty list for LAD temperatures
        avg_temp_list = []
        
        # For each LAD, creates boolean of which pixel in LAD, then averages temperatures for each LAD
        for index, polygon in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc=f"Averaging by {geography}"):
            # creates boolean mask for if points in gdf_london_temps are within each london_lsoas_lads_bng polygon
            mask = gdf_london_temps.within(polygon['geometry'])
            # calculates average pixel value (temp) where mask is true 
            avg_temp = gdf_london_temps.loc[mask, 'Pixel_Value'].mean()
            avg_temp_list.append(avg_temp)
        
        # Add the average temp values to df1
        dataframe.loc[:, 'avg_temp'] = avg_temp_list
        
        # Clean and save out
        #dataframe = dataframe.drop(['level_0', 'index'], axis=1)

        output_file_path = os.path.join(FIN_DIR,f'{today}_{geography}_temps')

        return {"output_data": dataframe, "output_file_path": output_file_path}
        
    # Save averaged data
    try:
        result = average_by_geog()
        london_avg_temps = result["output_data"]
        london_avg_temps.to_file(result["output_file_path"], driver='ESRI Shapefile')
    except Exception as e:
        import pdb
        print(f"Exception occurred: {e}")
        pdb.post_mortem()

    result = average_by_geog(geography='LSOA')
    london_avg_temps = result["output_data"]
    london_avg_temps.to_file(result["output_file_path"], driver='ESRI Shapefile')

    print("##################################################")
    print("3: Finished analyse.py")
    print("##################################################")





