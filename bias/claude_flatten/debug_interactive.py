#!/usr/bin/env python3
"""Interactive debug tool: select global + per-image params, view corrected image and model fit."""

import sys
sys.path.insert(0, '..')

import base64
import io
import numpy as np
import torch
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from pathlib import Path

from common import load
from registry import MODELS

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           url_base_pathname='/surfacedebug/')

SCRIPT_DIR = Path(__file__).parent
PARAMS_DIR = SCRIPT_DIR / 'params'
HOT_PIXELS = np.load(SCRIPT_DIR / 'hot_pixels.npy')


def scan_params(pattern):
    return sorted(p.name for p in PARAMS_DIR.glob(pattern))


def arr_to_b64(arr):
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_arr(s):
    return np.load(io.BytesIO(base64.b64decode(s)))


def load_and_prep(path):
    img_np = load(path).astype(np.float64)
    img_np[HOT_PIXELS[:, 0], HOT_PIXELS[:, 1]] = np.nan
    for c in range(img_np.shape[1]):
        col = img_np[:, c]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return img_np


def global_params_for_half(g, half, flat_idx):
    """Extract half-specific params from a global npz as a dict with unqualified keys."""
    suffix = f'_{half}'
    out = {}
    for k in g.files:
        if k.endswith(suffix):
            base = k[:-len(suffix)]
            v = g[k]
            out[base] = v[flat_idx] if base == 'cj' else v
    return out


def per_img_params_for_half(p, half):
    """Extract half-specific params (keys like slopes_top → slopes)."""
    suffix = f'_{half}'
    skip = {'b', 'bias', 's', 'fit_start', 'fit_stop'}  # non-model keys
    out = {}
    for k in p.files:
        if k.endswith(suffix):
            base = k[:-len(suffix)]
            if base in skip:
                continue
            out[base] = p[k]
    return out


def compute_results(per_img_name, glob_name):
    """Load params and image, return corrected image + predictions."""
    p = np.load(PARAMS_DIR / per_img_name, allow_pickle=True)
    g = np.load(PARAMS_DIR / glob_name, allow_pickle=True) if glob_name else None

    model_name = str(p['model_name'])
    ModelClass = MODELS[model_name]

    img_np = load_and_prep(str(p['image_path']))
    flat_idx = p['flat_idx']

    bias_full = np.empty_like(img_np)
    bias_full[:512] = p['bias_top']
    bias_full[512:] = p['bias_bot']
    img_corrected = img_np - bias_full

    preds = {}
    for half in ['top', 'bot']:
        r0 = int(p[f'fit_{half}_start'])
        r1 = int(p[f'fit_{half}_stop'])
        b = torch.tensor(p[f'b_{half}'], dtype=torch.float32)
        s = torch.tensor(p[f's_{half}'], dtype=torch.float32).unsqueeze(1)

        global_p = global_params_for_half(g, half, flat_idx) if g is not None else None
        per_img = per_img_params_for_half(p, half)

        m = ModelClass.from_params(b, s, global_p=global_p, per_img=per_img)
        with torch.no_grad():
            pred = m(b, s).numpy()

        img_corrected[r0:r1, flat_idx] = img_np[r0:r1][:, flat_idx] - pred
        preds[half] = pred

    return img_corrected, img_np, preds, p, model_name


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label('Per-image params:'),
            dcc.Dropdown(id='img-select', options=scan_params('nfi_fit_*.npz'),
                         value=(scan_params('nfi_fit_*.npz') or [None])[0], clearable=False),
            html.Label('Global params:', className='mt-2'),
            dcc.Dropdown(id='glob-select', options=scan_params('nfi_glob_*.npz'),
                         value=None, clearable=True),
            html.Div(id='model-info', className='mt-2 text-muted small'),
            html.Button('Refresh', id='refresh-btn', n_clicks=0,
                        className='btn btn-secondary btn-sm mt-2'),
        ], width=3),
        dbc.Col([
            dcc.Graph(id='image-plot', style={'height': '85vh'},
                      config={'scrollZoom': True}),
        ], width=4),
        dbc.Col([
            dcc.Graph(id='scatter-plot', style={'height': '85vh'}),
        ], width=5),
    ]),
    dcc.Store(id='results-store'),
], fluid=True)


@app.callback(
    Output('glob-select', 'options'),
    Output('glob-select', 'value'),
    Output('img-select', 'options'),
    Output('img-select', 'value'),
    Output('model-info', 'children'),
    Input('refresh-btn', 'n_clicks'),
    State('glob-select', 'value'),
    State('img-select', 'value'),
)
def refresh_options(_, cur_glob, cur_img):
    img_opts = scan_params('nfi_fit_*.npz')
    glob_opts = scan_params('nfi_glob_*.npz')
    img = cur_img if cur_img in img_opts else (img_opts[0] if img_opts else None)
    gval = cur_glob if cur_glob in glob_opts else None

    info = ''
    if img:
        p = np.load(PARAMS_DIR / img, allow_pickle=True)
        info = f'model: {str(p["model_name"])}'
    return glob_opts, gval, img_opts, img, info


@app.callback(
    Output('results-store', 'data'),
    Input('glob-select', 'value'),
    Input('img-select', 'value'),
)
def update_store(glob_name, img_name):
    if not img_name:
        return None

    img_corrected, img_np, preds, p, model_name = compute_results(img_name, glob_name)

    flat_idx = p['flat_idx'].tolist()
    return {
        'img_corrected': arr_to_b64(img_corrected),
        'img_np': arr_to_b64(img_np),
        'pred_top': arr_to_b64(preds['top']),
        'pred_bot': arr_to_b64(preds['bot']),
        'flat_idx': flat_idx,
        'fit_top_start': int(p['fit_top_start']),
        'fit_top_stop': int(p['fit_top_stop']),
        'fit_bot_start': int(p['fit_bot_start']),
        'fit_bot_stop': int(p['fit_bot_stop']),
        's_top': p['s_top'].tolist(),
        's_bot': p['s_bot'].tolist(),
    }


@app.callback(
    Output('image-plot', 'figure'),
    Input('results-store', 'data'),
)
def update_image(data):
    if not data:
        return go.Figure()
    img_corr = b64_to_arr(data['img_corrected'])
    fig = go.Figure(data=go.Heatmap(
        z=img_corr.astype(np.float16), colorscale='RdBu_r', zmid=0, zmin=-10, zmax=10,
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
    State('results-store', 'data'),
)
def update_scatter(click_data, data):
    if not click_data or not data:
        return go.Figure()

    col = int(round(click_data['points'][0]['x']))
    row = int(round(click_data['points'][0]['y']))
    half = 'top' if row < 512 else 'bot'

    fit_start = data[f'fit_{half}_start']
    fit_stop = data[f'fit_{half}_stop']
    s = np.array(data[f's_{half}'])
    flat_idx = data['flat_idx']

    img_np = b64_to_arr(data['img_np'])
    y_actual = img_np[fit_start:fit_stop, col]

    pred = b64_to_arr(data[f'pred_{half}'])
    if col in flat_idx:
        ch = flat_idx.index(col)
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
    State('results-store', 'data'),
    prevent_initial_call=True,
)
def update_crosshairs(hover_data, image_click, data):
    from dash import Patch
    patched = Patch()
    if not hover_data or not image_click or not data:
        patched['layout']['shapes'] = []
        return patched

    s_hovered = hover_data['points'][0]['x']
    col = int(round(image_click['points'][0]['x']))
    row_clicked = int(round(image_click['points'][0]['y']))
    half = 'top' if row_clicked < 512 else 'bot'

    s_arr = np.array(data[f's_{half}'])
    fit_start = data[f'fit_{half}_start']
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
