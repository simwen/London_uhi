import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
from shapely import wkt
import plotly.express as px
from scipy.stats import percentileofscore
import plotly.graph_objects as go

## 1. Plotly of temperature by LAD ----------------------------------------------
run = '2024-06-26'

## Load avg temp by geog (LAD)
# Plotly uses standard lat-longs so we change the crs
geo_file_path = os.path.join(DATA_DIR, f'{run}-1', 'final', f'{run}_LAD_temps',f'{run}_LAD_temps.shp')
london_avg_temps = gpd.read_file(geo_file_path)

london_avg_temps = london_avg_temps.set_crs('EPSG:27700')
london_temps_plotly = london_avg_temps.to_crs('EPSG:4326')

# The index of the dataframe is used as the choropleth labels
london_temps_plotly = london_temps_plotly.set_index('LAD22NM')

# london_temps_plotly = london_temps_plotly.round({'avg_temp':2})

# Function to create percentile of column
def create_percentile(df, col, new_name):
    df[f'{new_name}'] = df[f'{col}'].apply(
    lambda x: round(percentileofscore(df[f'{col}'], x, kind='rank'),0)
)
create_percentile(df = london_temps_plotly, col = 'avg_temp', new_name = 'perc_temp')

# Create an interactive temp map
fig = px.choropleth_mapbox(
    london_temps_plotly,
    geojson=london_temps_plotly.geometry,
    locations=london_temps_plotly.index,
    color='perc_temp',
    color_continuous_scale="matter",
    mapbox_style="carto-positron",
    center={"lat": london_temps_plotly.geometry.centroid.y.mean(), "lon": london_temps_plotly.geometry.centroid.x.mean()},
    zoom=8.2,
    opacity=0.95,
    labels={'perc_temp': 'Temperature percentile', '_index': "Local Authority"},
    title="London Boroughs' Average Temperature <br><br><sup>Interactive map - zoom & hover for detail</sup>"
)

# save the plot
fig.write_html(os.path.join(DATA_DIR, f'{run}-1', 'final', "temp_by_lad_map.html"))


## 2. Pixel plot ------------------------------------
ldn_outline = pd.read_csv(os.path.join(ALL_RUNS, 'london_outline_bng.csv'))
ldn_outline_geom = wkt.loads(ldn_outline['geometry'].iloc[0])  # Convert WKT to Shapely geometry

# Create a GeoDataFrame from the Shapely geometry
ldn_outline_geom = gpd.GeoDataFrame({'geometry': [ldn_outline_geom]}, crs='EPSG:27700')

gdf_london_temps = gpd.read_file(os.path.join(DATA_DIR, f'{run}-1', 'intermediate', f'shapefile {run}', "celcius_gdf.shp"))
gdf_london_temps = gdf_london_temps.set_crs('EPSG:27700')
gdf_london_temps_crop = gdf_london_temps[gdf_london_temps.within(ldn_outline_geom.iloc[0].geometry)]
gdf_london_temps_crop.to_file(os.path.join(DATA_DIR, f'{run}-1', 'intermediate', f'shapefile {run}', "celcius_gdf_crop.shp"))

gdf_london_temps_crop = gpd.read_file(os.path.join(DATA_DIR, f'{run}-1', 'intermediate', f'shapefile {run}', "celcius_gdf_crop.shp"))

gdf_london_temps_crop.loc[gdf_london_temps_crop['Pixel_Valu'] > 39, 'Pixel_Valu'] = 39
gdf_london_temps_crop.loc[gdf_london_temps_crop['Pixel_Valu'] < 19, 'Pixel_Valu'] = 19

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf_london_temps_crop.plot(column='Pixel_Valu', ax=ax, 
                            cmap='coolwarm', marker='o', markersize=5)

# remove axes/box
ax.set_axis_off()

# Define the formatter function
def format_axis_ticks(value, tick_number):
    return f'{int(value)}'

# Create the formatter
formatter = FuncFormatter(format_axis_ticks)

# Get the colorbar from the current figure and set the formatter
cbar = ax.get_figure().colorbar(ax.collections[0], ax=ax, orientation='vertical', shrink=0.6)
cbar.set_label("Temperature (°C)")
cbar.ax.yaxis.set_major_formatter(formatter)

plt.title('London Temperatures')
plt.savefig(os.path.join(DATA_DIR, f'{run}-1', 'final',f'{today}_pixel_temps2'))
#plt.show()

# Show the plot with temperature in degrees Celsius
with rio.open(os.path.join(DATA_DIR, f'{run}-1', 'intermediate',f'{run}_avg_celcius.tif')) as pixel_temps:

    print(pixel_temps.bounds)
    # Crop the raster to the polygon geometry
    cropped_data, cropped_transform = mask(pixel_temps, [ldn_outline_geom], crop=True)
    
    cropped_data = cropped_data.astype(float)

    # Create a mask for NaN values
    nan_mask = np.isnan(cropped_data)

    # Set values outside the polygon to NaN
    cropped_data[nan_mask] = np.nan

    plot = show(cropped_data, transform=cropped_transform, cmap='viridis', ax=plt.gca(), vmin=18)

    # Add a colorbar
    colorbar = plot.get_figure().colorbar(plot.get_images()[0], ax=plot.axes, orientation='vertical', pad=0.02)
    colorbar.set_label('Temperature (°C)')

    # Show the plot
    plt.show()
        

## 3. Correlation with deprivation ------------------------------------

# Load deprivation by lsoa
# Deprivation indices: b. Income Deprivation Domain & e. Health Deprivation and Disability Domain
## https://opendatacommunities.org/resource?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fimd2019%2Findices
lsoa_deprivation = pd.read_csv(os.path.join(ALL_RUNS,'imd2019lsoa.csv'))
lsoa_deprivation = lsoa_deprivation[lsoa_deprivation['Measurement'] == 'Score']
lsoa_deprivation = lsoa_deprivation[lsoa_deprivation['Indices of Deprivation'] == 'a. Index of Multiple Deprivation (IMD)']
lsoa_deprivation = lsoa_deprivation[lsoa_deprivation['DateCode'] == 2019]
lsoa_deprivation.rename(columns={'Value': 'multiple_deprivation'}, inplace=True)

lsoa_deprivation = lsoa_deprivation[['FeatureCode', 'multiple_deprivation']]


# Load 2021 LSOA lookup as lsoa_deprivation uses 2011 lsoa names/boundaries 
## https://geoportal.statistics.gov.uk/datasets/e99a92fb7607495689f2eeeab8108fd6/explore
boundary_lookup = pd.read_csv(os.path.join(ALL_RUNS,'boundary_change_lookups.csv'))
boundary_lookup = boundary_lookup[['LSOA11CD', 'LSOA21CD']]
lsoa_deprivation = pd.merge(lsoa_deprivation, boundary_lookup, how='right', left_on='FeatureCode', right_on='LSOA11CD')


# Join to london_lsoa_temps
london_lsoa_temps = gpd.read_file(os.path.join(DATA_DIR, f'{run}-1', 'final',f'{run}_LSOA_temps',f'{run}_LSOA_temps.shp'))
london_lsoas_dep = pd.merge(london_lsoa_temps, lsoa_deprivation, how='left', left_on='LSOA21CD', right_on='LSOA21CD')
# Fill in LSOAs with NA for deprivation with avg of surrounding LSOAs

# Create a spatial index for efficient spatial queries (a data structure that enables faster retrieval of spatial information)
# import pygeos
# london_lsoas_dep_sindex = london_lsoas_dep.sindex

# Create function to calculate average multiple_deprivation based on neighboring areas if nan
def calculate_average_multiple_deprivation(row):
    
    if np.isnan(row['multiple_deprivation']):
        # Get the bounding box of the current polygon
        bbox = row['geometry'].bounds

        # Use spatial index to find neighboring polygons
        possible_matches_index = list(london_lsoas_dep.intersection(bbox))
        possible_matches = london_lsoas_dep.iloc[possible_matches_index]

        # Filter out the current polygon
        possible_matches = possible_matches[possible_matches.intersects(row['geometry'])]

        # Calculate the average multiple_deprivation for neighboring areas
        if len(possible_matches) > 1:  # Exclude the current polygon itself
            avg_multiple_deprivation = possible_matches['multiple_deprivation'].mean()
            return avg_multiple_deprivation
        else:
            return row['multiple_deprivation']
    else:
        return row['multiple_deprivation']

# Apply the function to create the new column 'multiple_deprivation_avg'
london_lsoas_dep['multiple_deprivation_clean'] = london_lsoas_dep.apply(calculate_average_multiple_deprivation, axis=1)

## Average to LAD 
london_lads = london_avg_temps[['LAD22NM', 'geometry']].to_crs('EPSG:4326')

london_lads_dep = london_lsoas_dep.groupby('LAD22NM').agg({'avg_temp': 'mean', 'multiple_deprivation_clean': 'mean'})
london_lads_dep = pd.merge(london_lads_dep, london_lads, how = 'left', on = 'LAD22NM')
london_lads_dep = gpd.GeoDataFrame(london_lads_dep, geometry='geometry')

# Calculate correlation
correlation = london_lads_dep['multiple_deprivation_clean'].corr(london_lads_dep['avg_temp']).round(2)
print(f"Correlation between multiple deprivation and temperature is: {correlation}")

# create percentile and set index
create_percentile(df = london_lads_dep, col = 'multiple_deprivation_clean', new_name = 'perc_dep')
london_lads_dep = london_lads_dep.set_index('LAD22NM')

# create percentile and set index
create_percentile(df = london_lsoas_dep, col = 'multiple_deprivation_clean', new_name = 'perc_dep')
create_percentile(df = london_lsoas_dep, col = 'avg_temp', new_name = 'perc_temp')

## Create top 5 stat
top_bottom_num = 5

london_lsoas_dep_cols = london_lsoas_dep[['LSOA21CD', 'LSOA21NM', 'multiple_deprivation_clean','perc_dep', 'avg_temp', 'perc_temp']].sort_values(by='multiple_deprivation_clean', ascending = False)
most_deprived_temp = london_lsoas_dep_cols['avg_temp'].head(top_bottom_num).mean()
least_deprived_temp = london_lsoas_dep_cols['avg_temp'].tail(top_bottom_num).mean()

temp_diff = most_deprived_temp-least_deprived_temp
print(f'The {top_bottom_num} most deprived LSOAs were on average {temp_diff:.1f}°C hotter than the {top_bottom_num} least deprived')

## 4. Plotly of deprivation by LAD ----------------------------------------------

fig = px.choropleth_mapbox(
    london_lads_dep,
    geojson=london_lads_dep.geometry,
    locations=london_lads_dep.index,
    color='perc_dep',
    color_continuous_scale="matter",
    mapbox_style="carto-positron",
    center={"lat": london_lads_dep.geometry.centroid.y.mean(), "lon": london_lads_dep.geometry.centroid.x.mean()},
    zoom=8.2,
    opacity=0.95,
    labels={'perc_dep': 'Deprivation Percentile', '_index': "Local Authority"},
    title="London Borough Deprivation Levels <br><br><sup>Interactive map - zoom & hover for detail</sup>"
)

# save the plot
fig.write_html(os.path.join(DATA_DIR, f'{run}-1', 'final', "dep_by_lad_map.html"))


# 5. Temp vs deprivation scatter ------------------------------------------------

london_lads_dep = london_lads_dep.round({'avg_temp':1, 'multiple_deprivation_clean':1})

## INTERACTIVE SCATTER
fig = px.scatter(
    london_lads_dep,
    x="multiple_deprivation_clean", 
    y="avg_temp", 
    title='Avg Deprivation vs Temperature by London Borough <br><sup>Interactive chart - hover for detail</sup>',
    labels={'multiple_deprivation_clean': 'Avg Deprivation', 'avg_temp': 'Avg Temperature'},
    opacity=0.5,
    hover_name=london_lads_dep.index
)

# Line of best fit
x = london_lads_dep['multiple_deprivation_clean']
y = london_lads_dep['avg_temp']
coefficients = np.polyfit(x, y, 3)
polynomial = np.poly1d(coefficients)
x_fit = np.linspace(min(x), max(x), 100)
y_fit = polynomial(x_fit)

fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Trend line',
                         line=dict(color='red', dash='dash', width=2),
                         hoverinfo='skip'))

# formatting
fig.update_layout(
    paper_bgcolor='white',  # Background color
    plot_bgcolor='white',   # Plot area background color
    width=640,              
    height=450,
    xaxis_title='Avg Deprivation Index', 
    yaxis_title='Avg Temperature (°C)',
    xaxis = dict(showline=True, linecolor='black', linewidth=1),
    yaxis = dict(showline=True,  linecolor='black',  linewidth=1),
    showlegend = False
)
fig.update_traces(marker=dict(size=9, color = '#2F6F7A')) 
fig.show()
# save the plot
fig.write_html(os.path.join(DATA_DIR, f'{run}-1', 'final', "dep_temp_scatter.html"))



## STATIC SCATTER
plt.figure(figsize=(7, 4))
plt.scatter(london_lads_dep['multiple_deprivation_clean'], london_lads_dep['avg_temp'], alpha=0.15)
plt.plot(x_fit, y_fit, color='red', linestyle='dashed', linewidth=2, label='Trend line')

# Title and labels
plt.title('Avg Deprivation vs Temperature by London Borough')
plt.xlabel('Average Multiple Deprivation Index')
plt.ylabel('Average Temperature')

# Legend
plt.legend()

# Show the plot
plt.show()
plt.close()


