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
import torch
from torchvision.io import read_image
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import cv2

from Utils import get_datasets, encode_image
from ImageRetrieval import CentroidRetrieval
from NeuralFeatureExtractor import MobileNetFeatureExtractor
from ImagePipeline import ImagePipeline

print()
print("Dashboard running")
print()

df_small, feat_small, df_unlabeled, feat_unlabeled = get_datasets()

image_folder = "..\Images\Dashboard"
unlabel_folder = "..\Dataset\\train_set"
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg"))]
image_paths = [os.path.join(image_folder, img) for img in image_files]

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Classificazione", href="/")),
        dbc.NavItem(dbc.NavLink("Retrieval", href="/page-1")),
        dbc.NavItem(dbc.NavLink("Pulizia", href="/page-2")),
    ],
    brand="Visual Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
)

carosello = html.Div(
    [
        # Freccia sinistra
        html.Button(
            "<",
            id="prev-button",
            n_clicks=0,
            style={
                "font-size": "20px",
                "position": "absolute",
                "left": "5px",  # Fissa la posizione a sinistra
                "top": "50%",  # Centra verticalmente
                "transform": "translateY(-50%)",  # Allinea perfettamente
                "z-index": "10",
            },
            className="btn btn-primary btn-lg",
        ),

        # Contenitore per l'immagine
        html.Div(
            html.Img(
                id="image-display",
                src=encode_image(image_paths[0]),
                style={
                    "max-width": "400px",  # Massima larghezza
                    "max-height": "400px",  # Massima altezza
                    "object-fit": "contain",  # Mantiene le proporzioni senza tagliare
                },
            ),
            style={
                "width": "400px",  # Stessa larghezza dell'immagine
                "height": "400px",
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "position": "relative",
            },
        ),

        # Freccia destra
        html.Button(
            ">",
            id="next-button",
            n_clicks=0,
            style={
                "font-size": "20px",
                "position": "absolute",
                "right": "5px",  # Fissa la posizione a destra
                "top": "50%",  # Centra verticalmente
                "transform": "translateY(-50%)",
                "z-index": "10",
            },
            className="btn btn-primary btn-lg",
        ),

        # Store per il percorso dell'immagine
        dcc.Store(id="image-path-store", data=image_paths[0]),
    ],
    style={
        "display": "flex",
        "align-items": "center",
        "justify-content": "center",
        "position": "relative",  # Permette il posizionamento assoluto delle frecce
        "width": "450px",  # Assicura che le frecce rimangano dentro il carosello
    },
)


retrieval_da_prescelte = html.Div(
    [
        html.Div(
            [
                html.Div(
                    children=[
                        html.H3(
                            "Immagine originale",
                            style={
                                "text-align": "center",
                                "margin-top": "20px",
                                "margin-bottom": "20px",
                            },
                        ),
                        carosello,
                        html.Button(
                            "Conferma",
                            id="confirm-button",
                            n_clicks=0,
                            style={"font-size": "20px", "margin-top": "20px"},
                            className="btn btn-primary btn-lg",
                        ),
                    ],
                    style={"width": "40%", "text-align": "center", "padding": "20px"},
                ),
                # Colonna 2: Immagini selezionate
                html.Div(
                    children=[
                        html.H3("Immagini selezionate"),
                        html.Div(
                            id="show-retrieval",
                            style={
                                "display": "flex",
                                "justify-content": "center",
                                "flex-wrap": "wrap",
                                "width": "100%",
                            },
                        ),
                    ],
                    style={"width": "60%", "text-align": "center", "padding": "20px"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",  # Layout orizzontale per l'immagine e la tabella
                "justifyContent": "space-between",
                "gap": "20px",
                "width": "100%",
            },
        ),
    ]
)


retrieval_prescelte = html.Div(
    children=[
        html.H1(
            "Retrieval",
            className="text-center text-primary",
            style={"font-size": "3rem", "font-weight": "bold", "margin-bottom": "20px"},
        ),
        retrieval_da_prescelte,
    ]
)
tabella_classificazione = dash.dash_table.DataTable(
    id="output-table",
    columns=[
        {"name": "Posizione", "id": "posizione"},
        {"name": "ID", "id": "id"},
        {"name": "Nome", "id": "nome"},
        {"name": "Probabilità", "id": "probabilita"},
    ],
    data=[],
    style_table={
        "margin-top": "20px",
        "width": "60%",
        "margin-left": "auto",
        "margin-right": "auto",
    },
    style_header={
        "backgroundColor": "#007bff",
        "color": "white",
        "fontWeight": "bold",
        "textAlign": "center",
    },
    style_cell={
        "textAlign": "center",
        "padding": "10px",
        "border": "1px solid #ddd",
        "fontSize": "16px",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#f2f2f2"},
        {"if": {"column_id": "probabilita"}, "color": "#28a745", "fontWeight": "bold"},
    ],
)


btn_classificazione = html.Div(
    [
        html.Button(
            "Classifica",
            id="classificazione-button",
            n_clicks=0,
            style={"font-size": "20px", "margin-top": "20px"},
            className="btn btn-primary btn-lg",
        ),
        html.Button(
            "Pulisci e Classifica",
            id="clean-classify-button",
            n_clicks=0,
            style={"font-size": "20px", "margin-top": "20px", "margin-left": "20px"},
            className="btn btn-primary btn-lg",
        ),
    ],
    style={"text-align": "center", "margin-left": "-150px"},
)

btn_pulisci = html.Button(
    "Pulisci",
    id="clean-button",
    n_clicks=0,
    style={"font-size": "20px", "margin-top": "20px", "margin-left": "-100px"},
    className="btn btn-primary btn-lg",
)


classificazione = html.Div(
    children=[
        html.H1(
            "Classificazione",
            className="text-center text-primary",
            style={"font-size": "3rem", "font-weight": "bold", "margin-bottom": "20px"},
        ),
        html.Div(
            children=[
                html.Div(
                    children=[carosello, btn_classificazione],
                    style={"width": "50%", "padding": "20px", "textAlign": "center"},
                ),
                html.Div(
                    children=[tabella_classificazione],
                    style={
                        "width": "50%",
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderLeft": "2px solid #007bff",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "center",
                "alignItems": "center",
                "gap": "20px",
                "width": "100%",
            },
        ),
    ]
)


app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        navbar,
        html.Div(
            [
                html.Div(id="page-content"),
            ],
            style={"width": "80%", "margin": "auto"},
        ),
    ]
)

pulizia = html.Div(
    children=[
        html.H1(
            "Pulizia tramite pipeline",
            className="text-center text-primary",
            style={"font-size": "3rem", "font-weight": "bold", "margin-bottom": "20px"},
        ),
        html.Div(
            children=[
                html.Div(
                    children=[carosello, btn_pulisci],
                    style={"width": "50%", "padding": "20px", "textAlign": "center"},
                ),
                html.Div(
                    children=[
                        html.Div(
                            id="show-pulizia",
                            style={
                                "display": "flex",
                                "justify-content": "center",
                                "flex-wrap": "wrap",
                                "width": "100%",
                            },
                        )
                    ],
                    style={
                        "width": "50%",
                        "padding": "20px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "center",
                "alignItems": "center",
                "gap": "20px",
                "width": "100%",
            },
        ),
    ]
    
)


@app.callback(
    dash.dependencies.Output("page-content", "children"),
    [dash.dependencies.Input("url", "pathname")],
)
def display_page(pathname):
    if pathname == "/page-1":
        return retrieval_prescelte
    if pathname == "/page-2":
        return pulizia
    else:
        return classificazione


@app.callback(
    [Output("image-display", "src"), Output("image-path-store", "data")],
    [Input("prev-button", "n_clicks"), Input("next-button", "n_clicks")],
)
def update_image(prev_clicks, next_clicks):
    index = (next_clicks - prev_clicks) % len(image_paths)
    return encode_image(image_paths[index]), image_paths[index]


@app.callback(
    Output("show-retrieval", "children"),
    [Input("confirm-button", "n_clicks")],
    [State("image-path-store", "data")],
)
def update_mini_gallery(n_clicks, selected_image):

    if n_clicks > 0:

        df_image = pd.DataFrame({"Image": [selected_image], "Label": [0]})

        extractor = MobileNetFeatureExtractor()

        torch_image = read_image(selected_image).type(torch.float32).div(255)

        feat_image = [extractor.compute_features_single_image(torch_image)]

        k = 5
        retrieval = CentroidRetrieval(
            df_image, feat_image, df_unlabeled, feat_unlabeled, k=k, metric="cosine"
        )

        retrieval.retrieve_images()

        df = retrieval.df_unlabeled
        image_files = df.loc[df["Label"] != -1, "Image"]

        image_paths = [os.path.join(unlabel_folder, img) for img in image_files]

        return [
            html.Img(
                src=encode_image(img),
                style={"width": "300px", "height": "300px", "margin": "5px"},
            )
            for img in image_paths
        ]
    return []


@app.callback(
    Output("output-table", "data"),
    [Input("classificazione-button", "n_clicks"), Input("clean-classify-button", "n_clicks")],
    [State("image-path-store", "data")],
)
def combined_classifier(class_n_clicks, clean_n_clicks, selected_image):
    ctx = dash.callback_context
    if not ctx.triggered:
        return []

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id not in ["classificazione-button", "clean-classify-button"]:
        return []

    # Caricamento dei dati
    x_train = np.load(
        os.path.join("..", "Features", "features", "train_features_retrieval.npy")
    )
    y_train = np.load(
        os.path.join("..", "Features", "labels", "train_labels_retrieval.npy")
    )

    model = KNeighborsClassifier(
        n_neighbors=51, n_jobs=-1, weights="distance", metric="cosine"
    )
    model.fit(x_train, y_train)

    extractor = MobileNetFeatureExtractor()
    preprocessing = triggered_id == "clean-classify-button"

    classifier = ImagePipeline(model, extractor, preprocessing=False)
    top5 = classifier.predict_for_dashboard(selected_image, preprocessing=preprocessing)

    if top5 is None:
        return [{"posizione": "1", "id": "-1", "nome": "Unknown", "probabilita": "100%"}]

    # Creazione della tabella
    table_data = [
        {"posizione": i + 1, "id": idx, "nome": name, "probabilita": f"{prob * 100:.2f} %"}
        for i, (idx, name, prob) in enumerate(top5)
    ]

    return table_data

@app.callback(
    Output("show-pulizia", "children"),
    [Input("clean-button", "n_clicks")],
    [State("image-path-store", "data")],
)
def pulisciImmagine(n_clicks, selected_image):
    if n_clicks > 0:
        cleaned_image_path = os.path.join("..\Images", "tmp.jpg")
        img_not_processed_path = os.path.join("..\Images", "teapot.jpg")

        x_train = np.load(
            os.path.join("..", "Features", "features", "train_features_retrieval.npy")
        )
        y_train = np.load(
            os.path.join("..", "Features", "labels", "train_labels_retrieval.npy")
        )

        model = KNeighborsClassifier(
            n_neighbors=51, n_jobs=-1, weights="distance", metric="cosine"
        )
        model.fit(x_train, y_train)
        extractor = MobileNetFeatureExtractor()

        classifier = ImagePipeline(model, extractor, preprocessing=False)

        image = cv2.imread(selected_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cleaned_image = classifier.preprocess(image, (20, 70))

        if cleaned_image is None:
            return [
                html.Img(
                    src=encode_image(img_not_processed_path),
                    style={"margin": "5px"},
                )
            ]
        
        cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(cleaned_image_path, cleaned_image)

        return [
            html.Img(
                src=encode_image(cleaned_image_path),
                style={
                    "max-width": "100%",  # Larghezza massima rispetto al contenitore
                    "max-height": "100%",  # Altezza massima rispetto al contenitore
                    "object-fit": "contain",  # Mantiene le proporzioni senza ritagliare
                    "margin": "5px",  # Margine attorno all'immagine
                }

            )
        ]
    return []


if __name__ == "__main__":
    app.run_server(debug=True)


# connect to http://127.0.0.1:8050/
