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

# Server-side cache — avoids roundtripping large arrays through the browser
_cache = {}

SCRIPT_DIR = Path(__file__).parent
PARAMS_DIR = SCRIPT_DIR / 'params'
FITS_DIR = SCRIPT_DIR / 'fits'
HOT_PIXELS = np.load(SCRIPT_DIR / 'hot_pixels.npy')


def scan_params(pattern):
    return sorted(p.name for p in PARAMS_DIR.glob(pattern))


def scan_fits():
    return sorted(p.name for p in FITS_DIR.glob('*.npz')) if FITS_DIR.exists() else []


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
                         value=(scan_params('nfi_fit_*.npz') or [None])[0], clearable=False, maxHeight=600),
            html.Label('Global params:', className='mt-2'),
            dcc.Dropdown(id='glob-select', options=scan_params('nfi_glob_*.npz'),
                         value=None, clearable=True, maxHeight=600),
            html.Label('Standalone fit:', className='mt-2'),
            dcc.Dropdown(id='standalone-select', options=scan_fits(),
                         value=None, clearable=True, maxHeight=600),
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
    dcc.Store(id='meta-store'),
    dcc.Store(id='standalone-store'),
    dcc.Store(id='column-store'),
], fluid=True)


@app.callback(
    Output('glob-select', 'options'),
    Output('glob-select', 'value'),
    Output('img-select', 'options'),
    Output('img-select', 'value'),
    Output('standalone-select', 'options'),
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
    return glob_opts, gval, img_opts, img, scan_fits(), info


@app.callback(
    Output('standalone-store', 'data'),
    Input('standalone-select', 'value'),
)
def load_standalone(name):
    if not name:
        return None
    npz = np.load(FITS_DIR / name)
    return {
        'filename': name,
        's': npz['s'].tolist(),
        'sel_flat': npz['sel_flat'].tolist(),
        'fit_top_start': int(npz['fit_top_start']),
        'fit_top_stop': int(npz['fit_top_stop']),
        'fit_bot_start': int(npz['fit_bot_start']),
        'fit_bot_stop': int(npz['fit_bot_stop']),
    }


@app.callback(
    Output('column-store', 'data'),
    Input('image-plot', 'clickData'),
    State('standalone-store', 'data'),
    State('meta-store', 'data'),
    prevent_initial_call=True,
)
def extract_column(click_data, standalone, meta):
    if not click_data:
        return no_update
    col = int(round(click_data['points'][0]['x']))
    row = int(round(click_data['points'][0]['y']))
    half = 'top' if row < 512 else 'bot'
    if standalone:
        r0 = standalone[f'fit_{half}_start']
        r1 = standalone[f'fit_{half}_stop']
        s = np.array(standalone['s'])[r0:r1]
        npz = np.load(FITS_DIR / standalone['filename'], mmap_mode='r')
        y_actual = npz['actual'][r0:r1, col]
        col_fit = npz['fit'][r0:r1, col]
        y_fit = None if np.all(np.isnan(col_fit)) else col_fit.tolist()
    elif meta and 'img_np' in _cache:
        r0 = meta[f'fit_{half}_start']
        r1 = meta[f'fit_{half}_stop']
        s = np.array(meta[f's_{half}'])
        flat_idx = meta['flat_idx']
        y_actual = _cache['img_np'][r0:r1, col]
        pred = _cache[f'pred_{half}']
        y_fit = pred[:, flat_idx.index(col)].tolist() if col in flat_idx else None
    else:
        return no_update
    return {'half': half, 'col': col, 's': s.tolist(), 'y_actual': y_actual.tolist(), 'y_fit': y_fit}


@app.callback(
    Output('results-store', 'data'),
    Output('meta-store', 'data'),
    Input('glob-select', 'value'),
    Input('img-select', 'value'),
)
def update_store(glob_name, img_name):
    if not img_name:
        return None, None

    img_corrected, img_np, preds, p, model_name = compute_results(img_name, glob_name)

    flat_idx = p['flat_idx'].tolist()
    meta = {
        'flat_idx': flat_idx,
        'fit_top_start': int(p['fit_top_start']),
        'fit_top_stop': int(p['fit_top_stop']),
        'fit_bot_start': int(p['fit_bot_start']),
        'fit_bot_stop': int(p['fit_bot_stop']),
        's_top': p['s_top'].tolist(),
        's_bot': p['s_bot'].tolist(),
    }
    _cache['img_corrected'] = img_corrected
    _cache['img_np'] = img_np
    _cache['pred_top'] = preds['top']
    _cache['pred_bot'] = preds['bot']
    return {'ready': True}, meta


@app.callback(
    Output('image-plot', 'figure'),
    Input('results-store', 'data'),
    Input('standalone-store', 'data'),
)
def update_image(data, standalone):
    if standalone:
        npz = np.load(FITS_DIR / standalone['filename'], mmap_mode='r')
        img_corr = np.where(np.isnan(npz['fit']), npz['actual'], npz['actual'] - npz['fit'])
    elif data and 'img_corrected' in _cache:
        img_corr = _cache['img_corrected']
    else:
        return go.Figure()
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
    Input('column-store', 'data'),
)
def update_scatter(col_data):
    if not col_data:
        return go.Figure()
    half = col_data['half']
    col = col_data['col']
    s = np.array(col_data['s'])
    y_actual = np.array(col_data['y_actual'])
    y_pred = np.array(col_data['y_fit']) if col_data['y_fit'] is not None else None

    sort_idx = np.argsort(s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s, y=y_actual, mode='markers',
        marker=dict(size=3, opacity=1.0),
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
    State('meta-store', 'data'),
    prevent_initial_call=True,
)
def update_crosshairs(hover_data, image_click, meta):
    from dash import Patch
    patched = Patch()
    if not hover_data or not image_click or not meta:
        patched['layout']['shapes'] = []
        return patched

    s_hovered = hover_data['points'][0]['x']
    col = int(round(image_click['points'][0]['x']))
    row_clicked = int(round(image_click['points'][0]['y']))
    half = 'top' if row_clicked < 512 else 'bot'

    s_arr = np.array(meta[f's_{half}'])
    fit_start = meta[f'fit_{half}_start']
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
    # Runs in atch session 'plotly'. To restart:
    #   echo -n $'\x03' | atch push plotly
    #   printf 'cd /home/evan/sync/research/carruthers/bias/claude_flatten && python debug_interactive.py\n' | atch push plotly
    app.run(debug=False, host='localhost', port=8890)
