import dash
from dash import dcc
from dash import html
# https://dash.plotly.com/dash-core-components
# https://dash.plotly.com/dash-html-components

import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import os

from Utils import get_datasets, encode_image
from ImageRetrieval import CentroidRetrieval

print()
print("Dashboard running")
print()

df_small, feat_small, df_unlabeled, feat_unlabeled = get_datasets()

image_folder = "..\Images"
unlabel_folder = "..\Dataset\\train_set"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg'))][:3]
image_paths = [os.path.join(image_folder, img) for img in image_files]

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Classificazione", href="/")),
        dbc.NavItem(dbc.NavLink("Retrieval", href="/page-1")),
    ],
    brand="Visual Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
)

retrieval_da_prescelte = html.Div([

    html.Div([
        html.H3("Immagini selezionate"),
        html.Div(id='show-retrieval', style={'display': 'flex', 'justify-content': 'center', 'flex-wrap': 'wrap'})
    ], style={'text-align': 'center', 'margin-top': '20px'}),

    html.H3("Immagine originale", style={'text-align': 'center', 'margin-top': '20px'}),

    html.Div([
        html.Button("<", id='prev-button', n_clicks=0, style={'font-size': '20px', 'margin': '10px'}, className='btn btn-primary btn-lg'),
        html.Img(id='image-display', src=encode_image(image_paths[0]), style={'width': '400px', 'height': 'auto'}),
        dcc.Store(id='image-path-store', data=image_paths[0]),
        html.Button(">", id='next-button', n_clicks=0, style={'font-size': '20px', 'margin': '10px'}, className='btn btn-primary btn-lg')
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),

    html.Div([
        html.Button("Conferma", id='confirm-button', n_clicks=0, style={'font-size': '20px', 'margin-top': '20px'}, className='btn btn-primary btn-lg')
    ], style={'text-align': 'center'})

])

retrieval = html.Div(children=[
    html.H1(
        'Retrieval',
        className='text-center text-primary',
        style={'font-size': '3rem', 'font-weight': 'bold', 'margin-bottom': '20px'}
    ),
    retrieval_da_prescelte
])

classificazione = html.Div(children=[
    html.H1('Classificazione'),
])

app.layout = html.Div(children=[

    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div([
        html.Div(id='page-content'),
    ], style={'width': '80%', 'margin': 'auto'})

])

@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/page-1':
        return retrieval
    else:
        return classificazione

@app.callback(
    [Output('image-display', 'src'), Output('image-path-store', 'data')],
    [Input('prev-button', 'n_clicks'), Input('next-button', 'n_clicks')]
)
def update_image(prev_clicks, next_clicks):
    index = (next_clicks - prev_clicks) % len(image_paths)
    return encode_image(image_paths[index]), image_paths[index]

@app.callback(
    Output('show-retrieval', 'children'),
    [Input('confirm-button', 'n_clicks')],
    [State('image-path-store', 'data')]
)
def update_mini_gallery(n_clicks, selected_image):
    
    if n_clicks > 0:

        df_image = pd.DataFrame({"Image": [selected_image]*20, "Label": [1]*20})

        if selected_image == os.path.join('..', 'Images', 'train_059364.jpg'):
           feat_image = feat_small[0:20, :]
        elif selected_image == os.path.join('..', 'Images', 'train_089424.jpg'):
            feat_image = feat_small[20:40, :]
        else:
            feat_image = feat_small[250*20:251*20, :]

        k = 5
        retrieval = CentroidRetrieval(df_image, feat_image, df_unlabeled, feat_unlabeled, k=k, metric="cosine")
        retrieval.retrieve_images()

        df = retrieval.df_unlabeled
        image_files = df.loc[df["Label"] != -1, "Image"]

        image_paths = [os.path.join(unlabel_folder, img) for img in image_files]

        return [html.Img(src=encode_image(img), style={'width': '300px', 'height': '300px', 'margin': '5px'}) for img in image_paths]
    return []

if __name__ == '__main__':
    app.run_server(debug=True)


#connecs to http://127.0.0.1:8050/