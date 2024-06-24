# Import packages
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
import os
import geopandas as gpd

# Set directories
BASE_DIR = 'Z:\Resources\Personal\Simeon Wentzel\london_uhi_data'
DATA_DIR = os.path.join(BASE_DIR,'data')
run = '2024-06-20'

# Load the data
## Load avg temp by geog (LAD)
# Plotly uses standard lat-longs so we change the crs
geo_file_path = os.path.join(DATA_DIR, f'{run}-1', 'final', f'{run}_LAD_temps',f'{run}_LAD_temps.shp')
london_temps_plotly = gpd.read_file(geo_file_path).set_crs('EPSG:27700')

london_temps_plotly = london_temps_plotly.to_crs('EPSG:4326')

# The index of the dataframe is used as the choropleth labels
london_temps_plotly = london_temps_plotly.set_index('LAD22CD')

# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children=[html.H1(children='London Urban Heat Islands')]),
    dcc.Graph(figure=px.bar(london_temps_plotly, x='LAD22NM', y='avg_temp')),
    dcc.Graph(figure=px.choropleth_mapbox(
        london_temps_plotly,
        geojson=london_temps_plotly.geometry,
        locations=london_temps_plotly.index,
        color='avg_temp',
        color_continuous_scale="matter",
        mapbox_style="carto-positron",
        center={"lat": london_temps_plotly.geometry.centroid.y.mean(), "lon": london_temps_plotly.geometry.centroid.x.mean()},
        zoom=8.5,
        opacity=0.9,
        labels={'avg_temp': 'Average Temperature', '_index': ""},
        title='Average Temperature by LAD'
    ))
]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)