#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from urllib.parse import urlencode, parse_qs
import json

img = np.load('images/oob_nfi_l0.npy')[100:512]
dark_orig = np.load('images/dark_nfi_l0.npy')[100:512]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/surface/')

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='initialized', data=False),
    dbc.Row([
        dbc.Col([
            html.Label('a:'),
            dcc.Slider(id='slider-a', min=0, max=2, step=0.004, value=1.0, marks={i: str(i) for i in [0, 0.5, 1, 1.5, 2]}),
            dbc.Row([
                dbc.Col(dcc.Input(id='a-min', type='number', value=0, placeholder='Start', size='sm'), width=4),
                dbc.Col(dcc.Input(id='a-max', type='number', value=2, placeholder='Stop', size='sm'), width=4),
                dbc.Col(dcc.Input(id='a-steps', type='number', value=500, placeholder='Steps', size='sm'), width=4),
            ]),
            html.Br(),
            html.Label('b:'),
            dcc.Slider(id='slider-b', min=-10, max=10, step=0.04, value=0, marks={i: str(i) for i in range(-10, 11, 5)}),
            dbc.Row([
                dbc.Col(dcc.Input(id='b-min', type='number', value=-10, placeholder='Start', size='sm'), width=4),
                dbc.Col(dcc.Input(id='b-max', type='number', value=10, placeholder='Stop', size='sm'), width=4),
                dbc.Col(dcc.Input(id='b-steps', type='number', value=500, placeholder='Steps', size='sm'), width=4),
            ]),
            html.Br(),
            html.Label('Transform (use "dark", "a", "b", "img", "np"):'),
            dcc.Textarea(id='transform', value='dark = dark * a + b\ny = img - dark\ns = np.sum(img, axis=1)', style={'width': '100%', 'height': 100}),
            html.Br(),
            html.Label('Selected columns (comma-separated):'),
            dcc.Input(id='cols', value='120,121,122,123,124,125,126,127,128,129,130', style={'width': '100%'}),
            html.Br(),
            html.Br(),
            dbc.Button('Update Image Plot', id='update-2d-btn', color='secondary', size='sm'),
            html.Br(),
            html.Br(),
            dbc.Button('Generate Settings URL', id='copy-btn', color='primary', size='sm'),
            html.Br(),
            html.Br(),
            dcc.Input(id='settings-url', type='text', readOnly=True, style={'width': '100%', 'fontSize': '11px'})
        ], width=3),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='plot-3d', style={'height': '95vh'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='plot-2d', style={'height': '95vh'})
                ], width=6)
            ])
        ], width=9)
    ])
], fluid=True)

def compute_values(a, b, transform_code):
    """Helper function to compute dark, y, s from transform code"""
    dark = dark_orig.copy()
    try:
        namespace = {'dark': dark, 'a': a, 'b': b, 'img': img, 'np': np}
        exec(transform_code, namespace)
        dark = namespace.get('dark', dark)
        y = namespace.get('y', img - dark)
        s = namespace.get('s', np.sum(img, axis=1))
    except Exception as e:
        print(e)
    return dark, y, s

@app.callback(
    Output('plot-3d', 'figure'),
    Input('slider-a', 'value'),
    Input('slider-b', 'value'),
    Input('transform', 'value'),
    Input('cols', 'value')
)
def update_3d_plot(a, b, transform_code, cols_str):
    dark, y, s = compute_values(a, b, transform_code)

    try:
        selected_cols = [int(c.strip()) for c in cols_str.split(',')]
    except:
        selected_cols = [127, 128]

    fig_3d = go.Figure()
    for col in selected_cols:
        fig_3d.add_trace(go.Scatter3d(
            x=dark[:, col], y=s, z=y[:, col], mode='markers',
            marker=dict(size=2), name=f'col {col}'
        ))

    fig_3d.update_layout(
        scene=dict(xaxis_title='Dark', yaxis_title='Sum', zaxis_title='Image'),
        title='3D Scatter',
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision='constant'
    )

    return fig_3d

@app.callback(
    Output('plot-2d', 'figure'),
    Input('update-2d-btn', 'n_clicks'),
    State('slider-a', 'value'),
    State('slider-b', 'value'),
    State('transform', 'value'),
    prevent_initial_call=True
)
def update_2d_plot(n_clicks, a, b, transform_code):
    dark, y, s = compute_values(a, b, transform_code)

    fig_2d = go.Figure(data=go.Heatmap(
        z=y,
        colorscale='Viridis'
    ))
    fig_2d.update_layout(
        title='Y (2D)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision='constant'
    )

    return fig_2d

@app.callback(
    Output('settings-url', 'value'),
    Input('copy-btn', 'n_clicks'),
    State('slider-a', 'value'),
    State('slider-b', 'value'),
    State('transform', 'value'),
    State('cols', 'value'),
    State('a-min', 'value'),
    State('a-max', 'value'),
    State('a-steps', 'value'),
    State('b-min', 'value'),
    State('b-max', 'value'),
    State('b-steps', 'value'),
    prevent_initial_call=True
)
def copy_settings(n_clicks, a, b, transform, cols, a_min, a_max, a_steps, b_min, b_max, b_steps):
    params = {
        'a': a, 'b': b, 'transform': transform, 'cols': cols,
        'a_min': a_min, 'a_max': a_max, 'a_steps': a_steps,
        'b_min': b_min, 'b_max': b_max, 'b_steps': b_steps
    }
    url = f'https://copernicus.ece.illinois.edu/surface/?{urlencode(params)}'
    return url

@app.callback(
    Output('slider-a', 'value'),
    Output('slider-b', 'value'),
    Output('transform', 'value'),
    Output('cols', 'value'),
    Output('a-min', 'value'),
    Output('a-max', 'value'),
    Output('a-steps', 'value'),
    Output('b-min', 'value'),
    Output('b-max', 'value'),
    Output('b-steps', 'value'),
    Output('initialized', 'data'),
    Input('url', 'search'),
    State('initialized', 'data'),
    prevent_initial_call=False
)
def load_from_url(search, initialized):
    default_transform = 'dark = dark * a + b\ny = img - dark\ns = np.sum(img, axis=1)'
    if initialized or not search:
        return [1, 0, default_transform, '120,121,122,123,124,125,126,127,128,129,130', 0, 2, 500, -10, 10, 500, True]

    params = parse_qs(search.lstrip('?'))
    return [
        float(params.get('a', [1])[0]),
        float(params.get('b', [0])[0]),
        params.get('transform', [default_transform])[0],
        params.get('cols', ['120,121,122,123,124,125,126,127,128,129,130'])[0],
        float(params.get('a_min', [0])[0]),
        float(params.get('a_max', [2])[0]),
        int(params.get('a_steps', [500])[0]),
        float(params.get('b_min', [-10])[0]),
        float(params.get('b_max', [10])[0]),
        int(params.get('b_steps', [500])[0]),
        True
    ]

@app.callback(
    Output('slider-a', 'min'),
    Output('slider-a', 'max'),
    Output('slider-a', 'step'),
    Input('a-min', 'value'),
    Input('a-max', 'value'),
    Input('a-steps', 'value'),
    prevent_initial_call=True
)
def update_slider_a(a_min, a_max, a_steps):
    if a_min is None or a_max is None or a_steps is None or a_steps <= 0:
        return 0, 2, 0.004
    return a_min, a_max, (a_max - a_min) / a_steps

@app.callback(
    Output('slider-b', 'min'),
    Output('slider-b', 'max'),
    Output('slider-b', 'step'),
    Input('b-min', 'value'),
    Input('b-max', 'value'),
    Input('b-steps', 'value'),
    prevent_initial_call=True
)
def update_slider_b(b_min, b_max, b_steps):
    if b_min is None or b_max is None or b_steps is None or b_steps <= 0:
        return -10, 10, 0.04
    return b_min, b_max, (b_max - b_min) / b_steps

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8009)
