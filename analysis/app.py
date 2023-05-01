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

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


@app.callback(
    Output("model-solution", "children"),
    Input("input-select", "label"),
    prevent_initial_call=True,
)
def compute_model(label):
    children = []
    model = NetworkFlowModel(data[label])
    sol = model.solve()
    children.append(dcc.Graph(figure=sol.ivisualize(ret=True), id="network"))
    children.append(
        dash_table.DataTable(
            sol.demands.to_dict("records"),
            [{"name": i, "id": i} for i in sol.demands.columns],
        )
    )
    return html.Div(children)


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
        html.H1(children="Network Flow Model"),
        html.Div(
            children=[
                dbc.DropdownMenu(items, id="input-select", label="Select Data"),
                dcc.Loading(
                    html.Div(id="model-solution"),
                ),
            ]
        ),
        # html.Div(
        #     children=[
        #         html.H2("Input Data", id="input-data-header"),
        #         html.Div(id="input-data-table"),
        #     ]
        # ),
        # dash_table.DataTable(
        #     data[input_name].to_dict("records"),
        #     [{"name": i, "id": i} for i in data[input_name].columns],
        # ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
