#!/usr/bin/env python3
"""Interactive debug tool: click a half-column on the image, see its scatter + model fit."""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, Patch
import dash_bootstrap_components as dbc
from pathlib import Path

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           url_base_pathname='/surfacedebug/')

npz_dir = Path(__file__).parent


def scan_npz():
    return sorted(str(p.name) for p in npz_dir.glob('*.npz'))


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label('Result:'),
            dcc.Dropdown(id='npz-select', options=scan_npz(),
                         value=(scan_npz() or [None])[0], clearable=False),
            html.Button('Refresh', id='refresh-btn', n_clicks=0,
                        className='btn btn-secondary btn-sm mt-2'),
        ], width=2),
        dbc.Col([
            dcc.Graph(id='image-plot', style={'height': '85vh'},
                      config={'scrollZoom': True}),
        ], width=5),
        dbc.Col([
            dcc.Graph(id='scatter-plot', style={'height': '85vh'}),
        ], width=5),
    ]),
], fluid=True)


@app.callback(
    Output('npz-select', 'options'),
    Output('npz-select', 'value'),
    Input('refresh-btn', 'n_clicks'),
    State('npz-select', 'value'),
)
def refresh_options(_, current):
    options = scan_npz()
    value = current if current in options else (options[0] if options else None)
    return options, value


@app.callback(
    Output('image-plot', 'figure'),
    Input('npz-select', 'value'),
)
def update_image(npz_name):
    if not npz_name:
        return go.Figure()
    d = dict(np.load(npz_dir / npz_name, allow_pickle=True))
    img_corr = d['img_corrected']

    fig = go.Figure(data=go.Heatmap(
        z=img_corr, colorscale='RdBu_r', zmid=0, zmin=-10, zmax=10,
        hovertemplate='col: %{x}<br>row: %{y}<br>val: %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title='Corrected image (click a column)',
        yaxis=dict(scaleanchor='x', autorange='reversed'),
        margin=dict(l=0, r=0, t=40, b=0),
        dragmode='pan',
    )
    return fig


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('image-plot', 'clickData'),
    State('npz-select', 'value'),
)
def update_scatter(click_data, npz_name):
    if not click_data or not npz_name:
        return go.Figure()

    col = int(round(click_data['points'][0]['x']))
    row = int(round(click_data['points'][0]['y']))
    d = dict(np.load(npz_dir / npz_name, allow_pickle=True))
    flatten_idx = d['flatten_idx'] if 'flatten_idx' in d else d['oob_idx']

    half = 'top' if row < 512 else 'bot'

    if half == 'top':
        fit_start = int(d['fit_top_start'])
        fit_stop = int(d['fit_top_stop'])
        s = d['s_top']
        pred = d['pred_top']
    else:
        fit_start = int(d['fit_bot_start'])
        fit_stop = int(d['fit_bot_stop'])
        s = d['s_bot']
        pred = d['pred_bot']

    img = d['img']
    y_actual = img[fit_start:fit_stop, col]

    oob_list = flatten_idx.tolist()
    if col in oob_list:
        ch = oob_list.index(col)
        y_pred = pred[:, ch]
    else:
        y_pred = None

    sort_idx = np.argsort(s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s, y=y_actual, mode='markers',
        marker=dict(size=3, opacity=0.4),
        name='actual',
    ))
    if y_pred is not None:
        fig.add_trace(go.Scatter(
            x=s[sort_idx], y=y_pred[sort_idx], mode='lines',
            line=dict(color='red', width=2),
            name='model',
        ))

    fig.update_layout(
        title=f'{half} half, col {col}',
        xaxis_title='Row sum (s)',
        yaxis_title='Pixel value',
        margin=dict(l=40, r=0, t=40, b=40),
    )
    return fig


@app.callback(
    Output('image-plot', 'figure', allow_duplicate=True),
    Input('scatter-plot', 'hoverData'),
    State('image-plot', 'clickData'),
    State('npz-select', 'value'),
    prevent_initial_call=True,
)
def update_crosshairs(hover_data, image_click, npz_name):
    patched = Patch()
    if not hover_data or not image_click or not npz_name:
        patched['layout']['shapes'] = []
        return patched

    s_hovered = hover_data['points'][0]['x']
    col = int(round(image_click['points'][0]['x']))
    row_clicked = int(round(image_click['points'][0]['y']))
    half = 'top' if row_clicked < 512 else 'bot'

    d = dict(np.load(npz_dir / npz_name, allow_pickle=True))
    s_arr = d[f's_{half}']
    fit_start = int(d[f'fit_{half}_start'])
    abs_row = fit_start + int(np.argmin(np.abs(s_arr - s_hovered)))

    patched['layout']['shapes'] = [
        dict(type='line', x0=0, x1=1, xref='paper',
             y0=abs_row, y1=abs_row, yref='y',
             line=dict(color='yellow', width=1)),
        dict(type='line', x0=col, x1=col, xref='x',
             y0=0, y1=1, yref='paper',
             line=dict(color='yellow', width=1)),
    ]
    return patched


if __name__ == '__main__':
    app.run(debug=False, host='localhost', port=8890)
