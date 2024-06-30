# Import packages
from dash import Dash, html, dash_table, dcc, Input, Output
import pandas as pd
import plotly.express as px
import os
import geopandas as gpd

# Set directories
DATA_DIR = os.path.join(BASE_DIR,'data')
run = '2024-06-26'

# Load the data
## Load avg temp by geog (LAD) and change the crs (as Plotly uses standard lat-longs)
geo_file_path = os.path.join(DATA_DIR, f'{run}-1', 'final')
lad_temps_plotly = gpd.read_file(os.path.join(geo_file_path, f'{run}_LAD_temps', f'{run}_LAD_temps.shp'))
lsoa_temps_plotly = gpd.read_file(os.path.join(geo_file_path, f'{run}_LSOA_temps', f'{run}_LSOA_temps.shp'))

def change_crs(gdf, crs_before, crs_after):
    gdf = gdf.set_crs(crs_before)
    gdf = gdf.to_crs(crs_after)
    return(gdf)

lad_temps_plotly = change_crs(gdf = lad_temps_plotly,
                              crs_before = 'EPSG:27700',
                              crs_after = 'EPSG:4326')

lsoa_temps_plotly = change_crs(gdf = lsoa_temps_plotly,
                               crs_before = 'EPSG:27700',
                               crs_after = 'EPSG:4326')


# The index of the dataframe is used as the choropleth labels
lad_temps_plotly = lad_temps_plotly.set_index('LAD22CD')
lsoa_temps_plotly = lsoa_temps_plotly.set_index('LSOA21CD')

# Sort descending
lad_temps_plotly = lad_temps_plotly.sort_values(by='avg_temp', ascending=False)
lsoa_temps_plotly = lsoa_temps_plotly.sort_values(by='avg_temp', ascending=False)


# Initialize the app
app = Dash()

# Initial charts
fig=px.bar(lad_temps_plotly, 
           x='LAD22NM', 
           y='avg_temp',
           labels={'LAD22NM': 'Borough', 
                   'avg_temp': 'Avg Temperature'})
fig.update_yaxes(range=[25, 35])

fig2=px.choropleth_mapbox(
    lad_temps_plotly,
    geojson=lad_temps_plotly.geometry,
    locations=lad_temps_plotly.index,
    color='avg_temp',
    color_continuous_scale="matter",
    mapbox_style="carto-positron",
    center={"lat": lad_temps_plotly.geometry.centroid.y.mean(), "lon": lad_temps_plotly.geometry.centroid.x.mean()},
    zoom=8.5,
    opacity=0.9,
    labels={'avg_temp': 'Average Temperature', '_index': ""},
    title='Average Temperature by LAD'
)

# App layout
app.layout = [
    
    html.Div(children=[html.H1(children='London Urban Heat Islands')]),
    
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': borough, 'value': borough} for borough in lad_temps_plotly['LAD22NM']],
        value=[], # sets default boroughs to empty list so shows all
        multi=True,
        style={'width': '70%'} 
    ),
    
    html.Div(children=[
        dcc.Graph(id='bar-chart', figure=fig),
        dcc.Graph(
            id='choropleth-map',
            figure=fig2),
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'justifyContent': 'space-between',
        'alignItems': 'flex-start',
        'height': '700px',
        'width': '100%'  # Ensure the container uses the full width
    }),
    
    #dcc.Graph(id='bar-chart', figure = fig),

    #dcc.Graph(id='choropleth-map', figure = fig2, 
    #          style={'width': '70%', 'height': '600px', 'margin': '0 auto'}),
    
    html.Div(id='output-container')
]

# Callback to update the output container based on the dropdown selection
@app.callback(
    Output('bar-chart', 'figure'),
    Output('choropleth-map', 'figure'),
    Input('dropdown', 'value')
)
def update_charts(selected_boroughs):
    
    # Filter to list of selected boroughs in dropdown
    if isinstance(selected_boroughs, str):
        selected_boroughs = [selected_boroughs]
    elif not selected_boroughs:
        selected_boroughs = lad_temps_plotly['LAD22NM'].tolist()

    filtered_df = lad_temps_plotly[lad_temps_plotly['LAD22NM'].isin(selected_boroughs)]
    
    fig = px.bar(filtered_df, 
                 x='LAD22NM', 
                 y='avg_temp',
                 labels={'LAD22NM': 'Borough', 
                         'avg_temp': 'Avg Temperature'})
    fig.update_yaxes(range=[25, 35])
    
    fig2 = px.choropleth_mapbox(
        filtered_df,
        geojson=filtered_df.geometry,
        locations=filtered_df.index,
        color='avg_temp',
        color_continuous_scale="matter",
        mapbox_style="carto-positron",
        center={"lat": filtered_df.geometry.centroid.y.mean(), "lon": filtered_df.geometry.centroid.x.mean()},
        zoom=8.5,
        opacity=0.9,
        labels={'avg_temp': 'Average Temperature', '_index': ""},
        title='Average Temperature by LAD'
    )

    return fig, fig2

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8054)
    # app here: http://127.0.0.1:8054/


