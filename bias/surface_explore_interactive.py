#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, Patch, no_update
import dash_bootstrap_components as dbc
from urllib.parse import urlencode, parse_qs
import json

from common import load, rob_bias, mean_bias
from pathlib import Path

# Discover and load all images: images[(dir, stem)] -> ndarray
image_dirs = sorted(str(d) for d in Path('.').glob('images_*'))
image_types = sorted(set(p.stem for d in image_dirs for p in Path(d).glob('*.pkl')))
images = {(d, p.stem): load(str(p)) for d in image_dirs for p in Path(d).glob('*.pkl')}

DEFAULTS = dict(
    a=95, a_min=0, a_max=100, a_steps=100,
    b=100, b_min=0, b_max=100, b_steps=100,
    cols=[100, 120],
)

default_setups = {
    'Robust Bias': """\
bias = rob_bias(orig, 150, 150)
res = np.load('residual.npy')
""",
    'Mean Bias': """\
bias = mean_bias(orig, 150, 150)
res = np.load('residual.npy')
""",
}

default_scripts = {
    'z/x/s': """\
img = orig - bias
#x = orig[rows]
x = bias[rows]
x = (bias - res)[rows]
y = (img)[rows]
s = np.sum(orig, axis=1)[rows]


# limits (set with a & b sliders)
y = np.clip(y, -100, np.percentile(y, a))
img = np.clip(img, -100, np.percentile(img, a))
s = np.clip(s, s.min(), np.percentile(s, b))

# row statistic of opposite side
#x = np.sum(orig, axis=1, keepdims=True).repeat(1024, axis=1)[rows]
#x = np.clip(x, -100, np.percentile(x, a))
#y = np.clip(y, -100, np.percentile(y, 95))

# labels
labelx = 'b'
labely = 's'
labelz = 'z'
""",
'z/s/s\'': """\
img = orig - bias
x = np.sum(orig, axis=1, keepdims=True).repeat(1024, axis=1)[rows_opp]
y = (img)[rows]
# row statistic of opposite side
s = np.sum(orig, axis=1)[rows]


# limits (set with a & b sliders)
x = np.clip(x, -100, np.percentile(x, a))
y = np.clip(y, -100, np.percentile(y, 95))
img = np.clip(img, -100, np.percentile(img, 95))
s = np.clip(s, s.min(), np.percentile(s, b))

# labels
labelx = 's_opp'
labely = 's'
labelz = 'z'

""",
}


def make_namespace(img_dir, img_type):
    return {
        'np': np,
        'orig': images[(img_dir, img_type)].copy(),
        'dark': images[(img_dir, 'dark_nfi_l0')].copy(),
        'rob_bias': rob_bias,
        'mean_bias': mean_bias,
        'rows': slice(512, None),
        'rows_opp': slice(0, 512),
    }

# Global namespace shared between setup and plotting
namespace = make_namespace(image_dirs[0], 'oob_nfi_l0')
exec(next(iter(default_setups.values())), namespace)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/surface/')

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='initialized', data=False),
    dcc.Store(id='setup-executed', data=0),
    dcc.Store(id='cols'),
    dcc.Store(id='rows'),
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
                    html.Label('Bias:'),
                    dcc.Dropdown(id='setup-preset', options=list(default_setups.keys()),
                                 value=next(iter(default_setups.keys())), clearable=False, style={'marginBottom': '8px'}),
                    dcc.Textarea(id='setup', value=next(iter(default_setups.values())), style={'width': '100%', 'height': 200}),
                    html.Br(),
                    html.Label('Selected rows:'),
                    dcc.Input(id='rows-input', type='text', value='512:', debounce=True, size='sm', style={'width': '100%'}),
                    html.Label('Selected columns:'),
                    dcc.Input(id='cols-input', type='text', value='1:1', debounce=True, size='sm', style={'width': '100%'}),
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
                dbc.Tab(label='Plotting', children=[
                    html.Br(),
                    dcc.Dropdown(id='plotting-preset', options=list(default_scripts.keys()),
                                 value=next(iter(default_scripts.keys())), clearable=False, style={'marginBottom': '8px'}),
                    dcc.Textarea(id='plotting', value=next(iter(default_scripts.values())), style={'width': '100%', 'height': 400}),
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
        ], width=2),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='plot-3d', style={'height': '80vh'}, clear_on_unhover=True)
                ], width=7),
                dbc.Col([
                    dcc.Graph(id='plot-2d', figure=go.Figure(layout=dict(dragmode='pan')), style={'height': '80vh'})
                ], width=5)
            ])
        ], width=10)
    ])
], fluid=True)

@app.callback(
    Output('setup', 'value', allow_duplicate=True),
    Input('setup-preset', 'value'),
    prevent_initial_call=True
)
def load_setup_preset(key):
    return default_setups[key]

@app.callback(
    Output('plotting', 'value', allow_duplicate=True),
    Input('plotting-preset', 'value'),
    prevent_initial_call=True
)
def load_plotting_preset(key):
    return default_scripts[key]

@app.callback(
    Output('rows', 'data'),
    Input('rows-input', 'value'),
)
def update_rows(value):
    try:
        start, stop = value.split(':')
        start = int(start) if start.strip() else 0
        stop = int(stop) if stop.strip() else None
    except:
        start, stop = 0, 512
    SPLIT = 512
    opp_start = (start + SPLIT) % 1024
    opp_stop = ((stop or 1024) + SPLIT) % 1024 or None
    namespace['rows'] = slice(start, stop)
    namespace['rows_opp'] = slice(opp_start, opp_stop)
    return [start, stop]

@app.callback(
    Output('cols', 'data'),
    Input('cols-input', 'value'),
)
def update_cols(value):
    try:
        start, stop = value.split(':')
        return [int(start), int(stop)]
    except:
        return [0, 0]

@app.callback(
    Output('setup-executed', 'data'),
    Input('setup', 'value'),
    Input('img-type', 'value'),
    Input('img-dir', 'value')
)
def execute_setup(setup_code, img_type, img_dir):
    """Run setup code and update global namespace."""
    global namespace
    namespace = make_namespace(img_dir, img_type)
    exec(setup_code, namespace)
    return hash((setup_code, img_type, img_dir))

def compute_plot(a, b, plot_code):
    """Run plotting code on existing namespace to compute x, y, s."""
    ns = {**namespace, 'a': a, 'b': b}
    exec(plot_code, ns)
    x = ns.get('x', np.zeros((1, 1)))
    y = ns.get('y', np.zeros((1, 1)))
    s = ns.get('s', np.zeros((1, 1)))
    img = ns.get('img', np.zeros((1, 1)))
    y2 = ns.get('y2', None)
    labelx = ns.get('labelx', 'x')
    labely = ns.get('labely', 's')
    labelz = ns.get('labelz', 'y')
    return x, y, s, img, y2, labelx, labely, labelz

@app.callback(
    Output('plot-3d', 'figure'),
    Input('slider-a', 'value'),
    Input('slider-b', 'value'),
    Input('plotting', 'value'),
    Input('cols', 'data'),
    Input('rows', 'data'),
    Input('setup-executed', 'data')
)
def update_3d_plot(a, b, plot_code, col_range, _rows, _):
    x, y, s, img, y2, labelx, labely, labelz = compute_plot(a, b, plot_code)
    selected_cols = list(range(col_range[0], col_range[1] + 1))
    s_min = float(np.min(s))

    colors = go.Figure().layout.template.layout.colorway

    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode='markers',
        marker=dict(size=8, color='white', line=dict(width=2, color='black')),
        showlegend=False, name='_highlight'
    ))
    for i, col in enumerate(selected_cols):
        c = colors[i % len(colors)]
        rows = np.arange(x.shape[0])
        customdata = np.column_stack([
            rows + namespace['rows'].start,
            np.full(len(rows), col)
        ])
        fig_3d.add_trace(go.Scatter3d(
            x=x[:, col], y=s, z=y[:, col], mode='markers',
            marker=dict(size=2, color=c), name=f'col {col}',
            customdata=customdata, hovertemplate='x: %{x}<br>s: %{y}<br>y: %{z}<extra>%{fullData.name}<br>row %{customdata[0]}</extra>'
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
            xaxis_title=labelx,
            yaxis_title=labely,
            zaxis_title=labelz
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
    State('plotting', 'value'),
    State('cols', 'data'),
    prevent_initial_call=True
)
def update_2d_plot(n_clicks, a, b, plot_code, col_range):
    x, y, s, img, y2, labelx, labely, labelz = compute_plot(a, b, plot_code)

    fig_2d = go.Figure(data=go.Heatmap(z=img, colorscale='Viridis',
                                       hovertemplate='col: %{x}<br>row: %{y}<br>y: %{z}<extra></extra>'))
    fig_2d.update_layout(
        title='y (2D)',
        yaxis=dict(scaleanchor='x', scaleratio=1, autorange='reversed'),
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision='constant',
        dragmode='pan',
        shapes=col_boundary_shapes(col_range)
    )

    return fig_2d

@app.callback(
    Output('settings-url', 'value'),
    Input('copy-btn', 'n_clicks'),
    State('slider-a', 'value'),
    State('slider-b', 'value'),
    State('plotting', 'value'),
    State('cols', 'data'),
    State('a-min', 'value'),
    State('a-max', 'value'),
    State('a-steps', 'value'),
    State('b-min', 'value'),
    State('b-max', 'value'),
    State('b-steps', 'value'),
    State('img-type', 'value'),
    State('img-dir', 'value'),
    State('setup-preset', 'value'),
    State('plotting-preset', 'value'),
    prevent_initial_call=True
)
def copy_settings(n_clicks, a, b, transform, cols, a_min, a_max, a_steps, b_min, b_max, b_steps,
                  img_type, img_dir, setup_preset, plotting_preset):
    params = {
        'a': a, 'b': b, 'plotting': transform,
        'cols': f'{cols[0]}:{cols[1]}',
        'a_min': a_min, 'a_max': a_max, 'a_steps': a_steps,
        'b_min': b_min, 'b_max': b_max, 'b_steps': b_steps,
        'img_type': img_type, 'img_dir': img_dir,
        'setup_preset': setup_preset, 'plotting_preset': plotting_preset,
    }
    url = f'https://copernicus.ece.illinois.edu/surface/?{urlencode(params)}'
    return url

@app.callback(
    Output('slider-a', 'value'),
    Output('slider-b', 'value'),
    Output('plotting', 'value'),
    Output('cols-input', 'value'),
    Output('a-min', 'value'),
    Output('a-max', 'value'),
    Output('a-steps', 'value'),
    Output('b-min', 'value'),
    Output('b-max', 'value'),
    Output('b-steps', 'value'),
    Output('img-type', 'value'),
    Output('img-dir', 'value'),
    Output('setup-preset', 'value'),
    Output('plotting-preset', 'value'),
    Output('initialized', 'data'),
    Input('url', 'search'),
    State('initialized', 'data'),
    prevent_initial_call=False
)
def load_from_url(search, initialized):
    D = DEFAULTS
    default_img_type = 'oob_nfi_l0'
    default_img_dir = image_dirs[0]
    default_setup_preset = next(iter(default_setups.keys()))
    default_plotting_preset = next(iter(default_scripts.keys()))
    if initialized or not search:
        return [D['a'], D['b'], next(iter(default_scripts.values())), f"{D['cols'][0]}:{D['cols'][1]}",
                D['a_min'], D['a_max'], D['a_steps'],
                D['b_min'], D['b_max'], D['b_steps'],
                default_img_type, default_img_dir, default_setup_preset, default_plotting_preset,
                True]

    params = parse_qs(search.lstrip('?'))
    setup_preset = params.get('setup_preset', [default_setup_preset])[0]
    plotting_preset = params.get('plotting_preset', [default_plotting_preset])[0]
    return [
        float(params.get('a', [D['a']])[0]),
        float(params.get('b', [D['b']])[0]),
        params.get('plotting', [default_scripts.get(plotting_preset, next(iter(default_scripts.values())))])[0],
        params.get('cols', [f"{D['cols'][0]}:{D['cols'][1]}"])[0],
        float(params.get('a_min', [D['a_min']])[0]),
        float(params.get('a_max', [D['a_max']])[0]),
        int(params.get('a_steps', [D['a_steps']])[0]),
        float(params.get('b_min', [D['b_min']])[0]),
        float(params.get('b_max', [D['b_max']])[0]),
        int(params.get('b_steps', [D['b_steps']])[0]),
        params.get('img_type', [default_img_type])[0],
        params.get('img_dir', [default_img_dir])[0],
        setup_preset,
        plotting_preset,
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

def col_boundary_shapes(col_range):
    return [
        dict(type='line', x0=col_range[0], x1=col_range[0], y0=0, y1=1, yref='paper', layer='below', line=dict(color='blue', width=2)),
        dict(type='line', x0=col_range[1], x1=col_range[1], y0=0, y1=1, yref='paper', layer='below', line=dict(color='blue', width=2)),
    ]

@app.callback(
    Output('plot-2d', 'figure', allow_duplicate=True),
    Input('cols', 'data'),
    prevent_initial_call=True
)
def update_col_boundaries(col_range):
    patched = Patch()
    patched['layout']['shapes'] = col_boundary_shapes(col_range)
    return patched

@app.callback(
    Output('plot-2d', 'figure', allow_duplicate=True),
    Input('plot-3d', 'hoverData'),
    State('cols', 'data'),
    prevent_initial_call=True
)
def highlight_pixel(hover_data, col_range):
    if not hover_data:
        patched = Patch()
        patched['layout']['shapes'] = col_boundary_shapes(col_range)
        return patched
    row, col = hover_data['points'][0]['customdata']
    patched = Patch()
    patched['layout']['shapes'] = col_boundary_shapes(col_range) + [
        dict(type='line', x0=col, x1=col, y0=0, y1=1, yref='paper', line=dict(color='red', width=1)),
        dict(type='line', x0=0, x1=1, xref='paper', y0=row, y1=row, line=dict(color='red', width=1)),
    ]
    return patched

@app.callback(
    Output('plot-3d', 'figure', allow_duplicate=True),
    Input('plot-2d', 'hoverData'),
    State('slider-a', 'value'),
    State('slider-b', 'value'),
    State('plotting', 'value'),
    prevent_initial_call=True
)
def highlight_3d_point(hover_data, a, b, plot_code):
    patched = Patch()
    if not hover_data:
        patched['data'][0]['x'] = []
        patched['data'][0]['y'] = []
        patched['data'][0]['z'] = []
        return patched
    col = int(round(hover_data['points'][0]['x']))
    row_abs = int(round(hover_data['points'][0]['y']))
    x, y, s, img, y2, *_ = compute_plot(a, b, plot_code)
    row_start = namespace['rows'].start or 0
    row_rel = row_abs - row_start
    if 0 <= row_rel < x.shape[0] and 0 <= col < x.shape[1]:
        patched['data'][0]['x'] = [float(x[row_rel, col])]
        patched['data'][0]['y'] = [float(s[row_rel])]
        patched['data'][0]['z'] = [float(y[row_rel, col])]
    else:
        patched['data'][0]['x'] = []
        patched['data'][0]['y'] = []
        patched['data'][0]['z'] = []
    return patched

if __name__ == '__main__':
    app.run(debug=False, host='localhost', port=8889)
