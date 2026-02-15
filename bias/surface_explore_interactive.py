#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from urllib.parse import urlencode, parse_qs
import json

from common import load, rob_bias
from pathlib import Path

# Discover and load all images: images[(dir, stem)] -> ndarray
image_dirs = sorted(str(d) for d in Path('.').glob('images_*'))
image_types = sorted(set(p.stem for d in image_dirs for p in Path(d).glob('*.pkl')))
images = {(d, p.stem): load(str(p)) for d in image_dirs for p in Path(d).glob('*.pkl')}

DEFAULTS = dict(
    a=95, a_min=0, a_max=100, a_steps=500,
    b=100, b_min=0, b_max=100, b_steps=500,
    cols=[100, 120],
)

default_setup = """\
robbias = rob_bias(img, 150, 150)
"""

default_transform = """\
x = img[512:]
y = (img - robbias)[512:-100]
s = np.sum(img, axis=1)[512:-100]

# limits (set with a & b sliders)
y = np.clip(y, -100, np.percentile(y, a))
s = np.clip(s, s.min(), np.percentile(s, b))
"""


def make_namespace(img_dir, img_type):
    return {
        'np': np,
        'img': images[(img_dir, img_type)].copy(),
        'dark': images[(img_dir, 'dark_nfi_l0')].copy(),
        'rob_bias': rob_bias,
    }

# Global namespace shared between setup and transform
namespace = make_namespace(image_dirs[0], 'oob_nfi_l0')
exec(default_setup, namespace)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/surface/')

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='initialized', data=False),
    dcc.Store(id='setup-executed', data=0),
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Setup', children=[
                    html.Br(),
                    html.Label('Image:'),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='img-type', options=image_types, value='oob_nfi_l0', clearable=False), width=6),
                        dbc.Col(dcc.Dropdown(id='img-dir', options=image_dirs, value=image_dirs[0], clearable=False), width=6),
                    ]),
                    html.Br(),
                    html.Label('Setup:'),
                    dcc.Textarea(id='setup', value=default_setup, style={'width': '100%', 'height': 400}),
                    html.Br(),
                    html.Label('Selected columns:'),
                    dcc.RangeSlider(id='cols', min=0, max=next(iter(images.values())).shape[1] - 1, step=1, value=[1, 1],
                                    marks=None, tooltip={'placement': 'bottom', 'always_visible': True}),
                    html.Br(),
                    html.Br(),
                    dbc.Button('Update Image Plot', id='update-2d-btn', color='secondary', size='sm'),
                    html.Br(),
                    html.Br(),
                    dbc.Button('Generate Settings URL', id='copy-btn', color='primary', size='sm'),
                    html.Br(),
                    html.Br(),
                    dcc.Input(id='settings-url', type='text', readOnly=True, style={'width': '100%', 'fontSize': '11px'})
                ]),
                dbc.Tab(label='Transform', children=[
                    html.Br(),
                    html.Label('Transform:'),
                    dcc.Textarea(id='transform', value=default_transform, style={'width': '100%', 'height': 400}),
                    html.Br(),
                    html.Label('a:'),
                    dcc.Slider(id='slider-a', min=1, max=1, step=1, value=1),
                    dbc.Row([
                        dbc.Col(dcc.Input(id='a-min', type='number', value=1, placeholder='Start', size='sm'), width=4),
                        dbc.Col(dcc.Input(id='a-max', type='number', value=1, placeholder='Stop', size='sm'), width=4),
                        dbc.Col(dcc.Input(id='a-steps', type='number', value=1, placeholder='Steps', size='sm'), width=4),
                    ]),
                    html.Br(),
                    html.Label('b:'),
                    dcc.Slider(id='slider-b', min=1, max=1, step=1, value=1),
                    dbc.Row([
                        dbc.Col(dcc.Input(id='b-min', type='number', value=1, placeholder='Start', size='sm'), width=4),
                        dbc.Col(dcc.Input(id='b-max', type='number', value=1, placeholder='Stop', size='sm'), width=4),
                        dbc.Col(dcc.Input(id='b-steps', type='number', value=1, placeholder='Steps', size='sm'), width=4),
                    ]),
                ]),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='plot-3d', style={'height': '80vh'})
                ], width=7),
                dbc.Col([
                    dcc.Graph(id='plot-2d', style={'height': '80vh'})
                ], width=5)
            ])
        ], width=8)
    ])
], fluid=True)

@app.callback(
    Output('setup-executed', 'data'),
    Input('setup', 'value'),
    Input('img-type', 'value'),
    Input('img-dir', 'value')
)
def execute_setup(setup_code, img_type, img_dir):
    """Run setup code and update global namespace."""
    global namespace
    try:
        namespace = make_namespace(img_dir, img_type)
        exec(setup_code, namespace)
    except Exception as e:
        print(f'Setup error: {e}')
    return hash((setup_code, img_type, img_dir))

def compute_transform(a, b, transform_code):
    """Run transform code on existing namespace to compute x, y, s."""
    try:
        ns = {**namespace, 'a': a, 'b': b}
        exec(transform_code, ns)
        x = ns.get('x', np.zeros((1, 1)))
        y = ns.get('y', np.zeros((1, 1)))
        s = ns.get('s', np.zeros((1, 1)))
        y2 = ns.get('y2', None)
    except Exception as e:
        import traceback
        print(f'Transform error: {traceback.format_exc()}')
        x = np.zeros((1, 1))
        y = np.zeros((1, 1))
        s = np.sum(y, axis=1)
        y2 = None
    return x, y, s, y2

@app.callback(
    Output('plot-3d', 'figure'),
    Input('slider-a', 'value'),
    Input('slider-b', 'value'),
    Input('transform', 'value'),
    Input('cols', 'value'),
    Input('setup-executed', 'data')
)
def update_3d_plot(a, b, transform_code, col_range, _):
    x, y, s, y2 = compute_transform(a, b, transform_code)
    selected_cols = list(range(col_range[0], col_range[1] + 1))
    s_min = float(np.min(s))

    colors = go.Figure().layout.template.layout.colorway

    fig_3d = go.Figure()
    for i, col in enumerate(selected_cols):
        c = colors[i % len(colors)]
        rows = np.arange(x.shape[0])
        fig_3d.add_trace(go.Scatter3d(
            x=x[:, col], y=s, z=y[:, col], mode='markers',
            marker=dict(size=2, color=c), name=f'col {col}',
            customdata=rows, hovertemplate='x: %{x}<br>s: %{y}<br>y: %{z}<extra>%{fullData.name}<br>row %{customdata}</extra>'
        ))

    if y2 is not None:
        for i, col in enumerate(selected_cols):
            c = colors[i % len(colors)]
            fig_3d.add_trace(go.Scatter3d(
                x=[x[0, col]], y=[s_min], z=[float(y2[col])], mode='markers',
                marker=dict(size=5, color=c, line=dict(width=2, color='black')),
                showlegend=False
            ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='s',
            zaxis_title='y'
        ),
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
    x, y, s, y2 = compute_transform(a, b, transform_code)

    fig_2d = go.Figure(data=go.Heatmap(
        z=y,
        colorscale='Viridis'
    ))
    fig_2d.update_layout(
        title='y (2D)',
        yaxis=dict(scaleanchor='x', scaleratio=1, autorange='reversed'),
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
        'a': a, 'b': b, 'transform': transform,
        'col_start': cols[0], 'col_stop': cols[1],
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
    D = DEFAULTS
    if initialized or not search:
        return [D['a'], D['b'], default_transform, D['cols'],
                D['a_min'], D['a_max'], D['a_steps'],
                D['b_min'], D['b_max'], D['b_steps'], True]

    params = parse_qs(search.lstrip('?'))
    return [
        float(params.get('a', [D['a']])[0]),
        float(params.get('b', [D['b']])[0]),
        params.get('transform', [default_transform])[0],
        [int(params.get('col_start', [D['cols'][0]])[0]), int(params.get('col_stop', [D['cols'][1]])[0])],
        float(params.get('a_min', [D['a_min']])[0]),
        float(params.get('a_max', [D['a_max']])[0]),
        int(params.get('a_steps', [D['a_steps']])[0]),
        float(params.get('b_min', [D['b_min']])[0]),
        float(params.get('b_max', [D['b_max']])[0]),
        int(params.get('b_steps', [D['b_steps']])[0]),
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
        return DEFAULTS['a_min'], DEFAULTS['a_max'], (DEFAULTS['a_max'] - DEFAULTS['a_min']) / DEFAULTS['a_steps']
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
        return DEFAULTS['b_min'], DEFAULTS['b_max'], (DEFAULTS['b_max'] - DEFAULTS['b_min']) / DEFAULTS['b_steps']
    return b_min, b_max, (b_max - b_min) / b_steps

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8009)
