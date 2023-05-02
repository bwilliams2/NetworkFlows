# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, dash_table, Input, Output, State, MATCH, ctx, ALL
import dash_bootstrap_components as dbc

import plotly.express as px
import pandas as pd
from utils import NetworkFlowModel
from pathlib import Path

fpath = Path(__file__).parent.parent.joinpath("data/NetworkFlowProblem-Data.xlsx")

wb = pd.ExcelFile(fpath)
data = {}
for sheet in wb.sheet_names:
    data[sheet] = wb.parse(sheet)

items = []
n = 0
for k in data.keys():
    if "Input" in k:
        items.append(
            dbc.DropdownMenuItem(
                k, id={"type": "input-menu-item", "index": n}, n_clicks=0
            )
        )
        n += 1

input_name = "Input2"

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Network Flow Solutions"


@app.callback(
    [
        Output(
            "model-solution",
            "style",
        ),
        Output("network", "figure"),
        Output("demands-table", "data"),
        Output("demands-table", "columns"),
    ],
    Input("input-select", "label"),
    prevent_initial_call=True,
)
def compute_model(label):
    children = []
    model = NetworkFlowModel(data[label])
    sol = model.solve()
    demands = sol.demands
    demands = demands.loc[:, ~demands.columns.str.contains("id")]
    figure = sol.ivisualize(ret=True)
    figure.update_layout(width=1800)
    table_data = demands.to_dict("records")
    columns = [
        {
            "name": col.split("_")[1] if "_" in col else col,
            "id": col,
        }
        for col in demands.columns
    ]
    return {"display": "block"}, figure, table_data, columns


@app.callback(
    Output("input-select", "label"),
    Input({"type": "input-menu-item", "index": ALL}, "n_clicks"),
    State({"type": "input-menu-item", "index": ALL}, "children"),
    State({"type": "input-menu-item", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def display_output(value, children, ids):
    button_clicked = ctx.triggered_id
    for n, id_val in enumerate(ids):
        if id_val == button_clicked:
            label = children[n]
            break
    return label


app.layout = html.Div(
    children=[
        html.H1(children="Network Flow Model Solutions", style={"textAlign": "center"}),
        html.Div(
            children=[
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Select Input Data:",
                                            style={
                                                "marginLeft": "2em",
                                                "textAlign": "right",
                                                "marginTop": "2px",
                                                "display": "inline-block",
                                                "marginRight": "10px",
                                            },
                                        ),
                                        dbc.DropdownMenu(
                                            items,
                                            id="input-select",
                                            label="Select Data",
                                            style={
                                                "display": "inline-block",
                                                "verticalAlign": "top",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            justify="center",
                        ),
                    ),
                    style={"margin": "auto", "width": "500px"},
                ),
                dcc.Loading(
                    html.Div(
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="network", style={"width": "90vw"}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    html.Hr(
                                                        style={
                                                            "borderWidth": "0.3vh",
                                                            "width": "100%",
                                                        }
                                                    )
                                                ),
                                                html.H2("Demands", id="demands-header"),
                                                html.Div(
                                                    dash_table.DataTable(
                                                        id="demands-table",
                                                    ),
                                                    style={
                                                        "overflowX": "auto",
                                                        "maxWidth": "100%",
                                                    },
                                                ),
                                            ],
                                            style={"margin": "2em"},
                                        ),
                                    ],
                                ),
                            ),
                            id="model-solution",
                            style={"display": "none"},
                        ),
                        style={"margin": "2em", "minHeight": "20vh", "margin": "1em"},
                    ),
                ),
            ]
        ),
        # html.Div(
        #     children=[
        #         html.H2("Input Data", id="input-data-header"),
        #         html.Div(id="input-data-table"),
        #     ],
        #     id="input-data-container",
        # ),
        # dash_table.DataTable(
        #     data[input_name].to_dict("records"),
        #     [{"name": i, "id": i} for i in data[input_name].columns],
        # ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
