#!/usr/bin/env python3
# Interactive Bokeh app for image subtraction and column analysis
# Run with: bokeh serve subtract_interactive.py --show --allow-websocket-origin=*

import json
import math
import numpy as np
from pathlib import Path
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import (ColumnDataSource, Slider, Spinner, MultiChoice, Select,
                          TextInput, Div, LinearColorMapper, DataRange1d,
                          Range1d, CustomJS, Button, LinearAxis, CheckboxGroup,
                          HoverTool)
from bokeh.layouts import row, column
from bokeh.palettes import Viridis256, Category10_10

from common import load, mean_bias

DEFAULT_GROUPS = [[0, 0.25], [0.25, 0.75], [0.75, 1]]
IMAGE_FOLDERS = sorted([p.name for p in Path('.').glob('images_*') if p.is_dir()])
ALL_FILES = {}  # folder -> list of stems
for _folder in IMAGE_FOLDERS:
    ALL_FILES[_folder] = [f.stem for f in sorted(Path(_folder).glob('*.pkl'))]
FILE_OPTIONS = sorted(set(f for flist in ALL_FILES.values() for f in flist))

SLIDER_DEFS = [
    # (name, default, lo, hi, steps, spinner_step)
    ('a', 80.0, 0.0, 500.0, 500, 1),
    ('b', 1.0, 0.0, 1.0, 500, 0.01),
    ('c', 1.0, 0.0, 1.0, 500, 0.01),
]


# --- Parse URL query params ---
_raw_args = {}
if curdoc().session_context and curdoc().session_context.request:
    _raw_args = curdoc().session_context.request.arguments


def _get(key, default):
    if key in _raw_args:
        try:
            raw = _raw_args[key][0].decode()
            if isinstance(default, float):
                return float(raw)
            elif isinstance(default, int):
                return int(float(raw))
            return raw
        except (ValueError, IndexError):
            pass
    return default


def _step(lo, hi, steps):
    return (hi - lo) / max(steps, 1)


def _format(step):
    if step <= 0 or step >= 1:
        return "0[.]0"
    d = min(math.ceil(-math.log10(step)), 6)
    return "0." + "0" * d


# --- Helper: slider group (slider + lo/hi/steps spinners) ---
def _make_slider_group(title, value, lo, hi, steps, spinner_step):
    s = _step(lo, hi, steps)
    slider = Slider(title=title, value=value,
                    start=lo, end=hi,
                    step=s, format=_format(s), width=250)
    w_lo = Spinner(title="Start", value=lo, step=spinner_step, width=80)
    w_hi = Spinner(title="Stop", value=hi, step=spinner_step, width=80)
    w_steps = Spinner(title="Steps", value=steps, step=1, low=1, width=80)
    return dict(slider=slider, lo=w_lo, hi=w_hi, steps=w_steps)


def _update_slider_range(sg):
    """Recompute step from lo/hi/steps spinners and update the slider."""
    step = _step(sg['lo'].value, sg['hi'].value, sg['steps'].value)
    sg['slider'].start = sg['lo'].value
    sg['slider'].end = sg['hi'].value
    sg['slider'].step = step
    sg['slider'].format = _format(step)


# --- Helper: collapsible toggle ---
def _make_collapsible(label):
    col = column(visible=False)
    btn = Button(label=f"\u25b6 {label}", width=300)
    btn.js_on_click(CustomJS(args=dict(col=col, btn=btn), code="""
        col.visible = !col.visible;
        btn.label = col.visible ? "\u25bc """ + label + """" : "\u25b6 """ + label + """";
    """))
    return btn, col


# --- Load images once ---
images = {}
for _folder in IMAGE_FOLDERS:
    for _f in ALL_FILES[_folder]:
        images[(_folder, _f)] = load(f'{_folder}/{_f}.pkl')


# --- Computation ---
def compute(folder, file_name, clip_level, row_agg_str, func_str, params, col_groups):
    x = images[(folder, file_name)].copy()
    ns = {'x': x, 'np': np, 'mean_bias': mean_bias, **params}
    exec(row_agg_str, ns)
    s = ns.get('s')
    exec(func_str, ns)
    x = ns['x']

    nrows, ncols = x.shape
    groups = []
    for frac_lo, frac_hi in col_groups:
        c0 = int(round(frac_lo * ncols))
        c1 = int(round(frac_hi * ncols))
        groups.append(np.median(x[:, c0:c1], axis=1))
    squish = np.stack(groups, axis=-1)

    stat = s.flatten() if s is not None and s.size > 0 else None
    return np.clip(squish, 0, clip_level), np.clip(x, 0, clip_level), stat


def make_line_data(squish, col_groups):
    n = len(col_groups)
    xs = [np.arange(squish.shape[0]).tolist() for _ in range(n)]
    ys = [squish[:, i].tolist() for i in range(n)]
    colors = list(Category10_10[:n])
    labels = [str(i) for i in range(n)]
    return dict(xs=xs, ys=ys, colors=colors, labels=labels)


def make_vline_data(ncols, nrows, col_groups):
    xs, ys = [], []
    for frac_lo, frac_hi in col_groups:
        for xpos in (frac_lo * ncols, frac_hi * ncols):
            xs.append([xpos, xpos])
            ys.append([0, nrows])
    return dict(xs=xs, ys=ys)


# --- Helper: per-image state ---
def _make_per_image(fname):
    clip_val = _get(f'clip.{fname}', 25.0)
    groups_val = _get(f'groups.{fname}', json.dumps(DEFAULT_GROUPS))
    clip_lo_val = _get(f'clip_lo.{fname}', 0.0)
    clip_hi_val = _get(f'clip_hi.{fname}', 10000.0)
    clip_steps_val = _get(f'clip_steps.{fname}', 500)

    try:
        json.loads(groups_val)
    except (json.JSONDecodeError, TypeError):
        groups_val = json.dumps(DEFAULT_GROUPS)

    w_groups = TextInput(value=groups_val, width=300)
    clip = _make_slider_group("Clip", clip_val, clip_lo_val, clip_hi_val,
                              clip_steps_val, 1)

    mapper = LinearColorMapper(palette=Viridis256, low=0, high=clip_val)
    line_src = ColumnDataSource(dict(xs=[], ys=[], colors=[], labels=[]))
    stat_src = ColumnDataSource(dict(x=[], y=[]))
    img_src = ColumnDataSource(dict(image=[], dw=[], dh=[]))
    vline_src = ColumnDataSource(dict(xs=[], ys=[]))

    fig_lines = figure(title=f"{fname} \u2014 Row Median",
                       y_range=Range1d(),
                       sizing_mode='stretch_both', min_height=200,
                       tools="pan,wheel_zoom,box_zoom,reset,save")
    fig_lines.multi_line(xs='xs', ys='ys', source=line_src,
                         line_color='colors', line_width=1.5,
                         legend_field='labels')
    fig_lines.legend.click_policy = "hide"

    fig_lines.extra_y_ranges = {"stat": Range1d()}
    fig_lines.add_layout(LinearAxis(y_range_name="stat",
                                    axis_label="Row Stat."), 'right')
    stat_line = fig_lines.line(x='x', y='y', source=stat_src,
                               y_range_name="stat",
                               line_color='gray', line_width=1.5, alpha=0.7,
                               visible=True, legend_label="Row Stat.")

    fig_img = figure(title=fname,
                     x_range=DataRange1d(range_padding=0),
                     y_range=DataRange1d(range_padding=0),
                     match_aspect=True,
                     sizing_mode='scale_height', width=400, height=400,
                     tools="pan,wheel_zoom,box_zoom,reset,save")
    fig_img.y_range.flipped = True
    img_renderer = fig_img.image(image='image', source=img_src, x=0, y=0, dw='dw', dh='dh',
                                 color_mapper=mapper)
    fig_img.add_tools(HoverTool(renderers=[img_renderer], tooltips=[
        ("x", "$x{0}"), ("y", "$y{0}"), ("value", "@image"),
    ]))
    fig_img.multi_line(xs='xs', ys='ys', source=vline_src,
                       line_color='red', line_width=2)

    return dict(
        w_groups=w_groups, clip=clip,
        line_src=line_src, stat_src=stat_src,
        img_src=img_src, vline_src=vline_src,
        fig_lines=fig_lines, fig_img=fig_img,
        mapper=mapper, stat_line=stat_line,
    )


# --- Per-image state (pre-create for all images) ---
rows = {fname: _make_per_image(fname) for fname in FILE_OPTIONS}


# --- Shared Widgets ---
_init_files_str = _get('files', 'oob_wfi')
_init_files = [f.strip() for f in _init_files_str.split(',')
               if f.strip() in FILE_OPTIONS]
if not _init_files:
    _init_files = [FILE_OPTIONS[0]] if FILE_OPTIONS else []

_init_folder = _get('folder', IMAGE_FOLDERS[0] if IMAGE_FOLDERS else '')
w_folder = Select(options=IMAGE_FOLDERS, value=_init_folder, width=300)
w_multichoice = MultiChoice(options=FILE_OPTIONS, value=_init_files, width=300)
w_plot_stat = CheckboxGroup(labels=["Plot Row Statistic"], active=[0])

w_row_agg = TextInput(title="Row Statistic",
                      value=_get('row_agg', 's = np.mean(x, axis=1, keepdims=True)'),
                      width=300)
w_func = TextInput(title="Function",
                   value=_get('func', 'x = x'), width=300)

# Build shared slider groups from SLIDER_DEFS
sliders = {}
for _name, _default, _lo, _hi, _steps, _ss in SLIDER_DEFS:
    sliders[_name] = _make_slider_group(
        _name,
        _get(_name, _default),
        _get(f'{_name}_lo', _lo),
        _get(f'{_name}_hi', _hi),
        _get(f'{_name}_steps', _steps),
        _ss,
    )

w_err = Div(text="", width=300)


# --- Dynamic layout containers ---
w_per_image_toggle, per_image_col = _make_collapsible("Image Controls")
w_subtraction_toggle, subtraction_col = _make_collapsible("Subtraction")
plot_col = column(sizing_mode='stretch_both')


# --- Copy Settings button (client-side JS) ---
_url_args = dict(
    w_multichoice=w_multichoice,
    w_row_agg=w_row_agg,
    w_func=w_func,
)
for _name, *_ in SLIDER_DEFS:
    sg = sliders[_name]
    _url_args[f'w_{_name}'] = sg['slider']
    _url_args[f'w_{_name}_lo'] = sg['lo']
    _url_args[f'w_{_name}_hi'] = sg['hi']
    _url_args[f'w_{_name}_steps'] = sg['steps']

for _fname in FILE_OPTIONS:
    _r = rows[_fname]
    _url_args[f'g_{_fname}'] = _r['w_groups']
    _url_args[f'cl_{_fname}'] = _r['clip']['slider']
    _url_args[f'cllo_{_fname}'] = _r['clip']['lo']
    _url_args[f'clhi_{_fname}'] = _r['clip']['hi']
    _url_args[f'clst_{_fname}'] = _r['clip']['steps']

_pi_entries = []
for _fname in FILE_OPTIONS:
    _pi_entries.append(
        f"'{_fname}': {{g: g_{_fname}, cl: cl_{_fname}, "
        f"cllo: cllo_{_fname}, clhi: clhi_{_fname}, clst: clst_{_fname}}}")
_pi_js = '{\n        ' + ',\n        '.join(_pi_entries) + '\n    }'

# Data-driven shared slider JS lines
_slider_js_lines = []
for _name, *_ in SLIDER_DEFS:
    for _suffix in ('', '_lo', '_hi', '_steps'):
        _slider_js_lines.append(
            f"    p.set('{_name}{_suffix}', String(w_{_name}{_suffix}.value));")
_slider_js = '\n'.join(_slider_js_lines)

w_copy = Button(label="Copy Settings", width=150)
w_copy.js_on_click(CustomJS(args=_url_args, code=f"""
    const files = {json.dumps(FILE_OPTIONS)};
    const selected = w_multichoice.value;
    const pi = {_pi_js};
    const p = new URLSearchParams();

    p.set('files', selected.join(','));
    p.set('row_agg', String(w_row_agg.value));
    p.set('func', String(w_func.value));
{_slider_js}

    for (const fname of files) {{
        const r = pi[fname];
        p.set('groups.' + fname, String(r.g.value));
        p.set('clip.' + fname, String(r.cl.value));
        p.set('clip_lo.' + fname, String(r.cllo.value));
        p.set('clip_hi.' + fname, String(r.clhi.value));
        p.set('clip_steps.' + fname, String(r.clst.value));
    }}

    const url = window.location.origin + window.location.pathname + '?' + p.toString();
    navigator.clipboard.writeText(url);
    cb_obj.label = "Copied!";
    setTimeout(() => {{ cb_obj.label = "Copy Settings"; }}, 1500);
"""))


# --- Server callbacks ---
def _get_params():
    return {name: sliders[name]['slider'].value for name, *_ in SLIDER_DEFS}


def refresh_image(fname):
    r = rows[fname]
    try:
        col_groups = json.loads(r['w_groups'].value)
        s, im, stat = compute(w_folder.value, fname, r['clip']['slider'].value,
                              w_row_agg.value, w_func.value,
                              _get_params(), col_groups)
        nr, nc = im.shape
        line_data = make_line_data(s, col_groups)
        all_ys = [v for sub in line_data['ys'] for v in sub]
        if all_ys:
            lo, hi = min(all_ys), max(all_ys)
            pad = max((hi - lo) * 0.05, 1e-6)
            r['fig_lines'].y_range.start = lo - pad
            r['fig_lines'].y_range.end = hi + pad
        r['line_src'].data = line_data

        # Update stat line
        if stat is not None:
            r['stat_src'].data = dict(x=np.arange(len(stat)).tolist(),
                                      y=stat.tolist())
            stat_lo, stat_hi = stat.min(), stat.max()
            stat_pad = max((stat_hi - stat_lo) * 0.05, 1e-6)
            r['fig_lines'].extra_y_ranges['stat'].start = stat_lo - stat_pad
            r['fig_lines'].extra_y_ranges['stat'].end = stat_hi + stat_pad
        else:
            r['stat_src'].data = dict(x=[], y=[])

        r['vline_src'].data = make_vline_data(nc, nr, col_groups)
        r['mapper'].high = max(r['clip']['slider'].value, 1e-6)
        r['img_src'].data = dict(image=[im], dw=[nc], dh=[nr])
        w_err.text = ""
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        w_err.text = f'<span style="color: red">{traceback.format_exc()}</span>'


def refresh_all():
    for fname in w_multichoice.value:
        refresh_image(fname)


def rebuild():
    selected = w_multichoice.value

    # Rebuild per-image controls in the left column
    children = []
    if selected:
        children.append(w_plot_stat)
        children.append(Div(text="<b>Column Grouping</b>"))
        for fname in selected:
            children.append(rows[fname]['w_groups'])
        for fname in selected:
            r = rows[fname]
            children.extend([
                r['clip']['slider'],
                row(r['clip']['lo'], r['clip']['hi'], r['clip']['steps']),
            ])
    per_image_col.children = children

    # Rebuild plot rows (one row per selected image)
    plot_children = []
    for fname in selected:
        r = rows[fname]
        plot_children.append(
            row(r['fig_lines'], r['fig_img'], sizing_mode='stretch_both'))
    plot_col.children = plot_children

    refresh_all()


def on_shared_change(attr, old, new):
    refresh_all()


def on_folder_change(attr, old, new):
    rebuild()


def on_multichoice_change(attr, old, new):
    rebuild()


def on_plot_stat_change(attr, old, new):
    show_stat = 0 in w_plot_stat.active
    for fname in w_multichoice.value:
        rows[fname]['stat_line'].visible = show_stat


# --- Per-image widget callbacks ---
for _fname in FILE_OPTIONS:
    _r = rows[_fname]

    def _make_img_cb(fn):
        def cb(attr, old, new):
            if fn in w_multichoice.value:
                refresh_image(fn)
        return cb

    _r['w_groups'].on_change('value', _make_img_cb(_fname))
    _r['clip']['slider'].on_change('value_throttled', _make_img_cb(_fname))

    def _make_range_cb(fn):
        r = rows[fn]
        def cb(attr, old, new):
            _update_slider_range(r['clip'])
            if fn in w_multichoice.value:
                refresh_image(fn)
        return cb

    _rcb = _make_range_cb(_fname)
    _r['clip']['lo'].on_change('value', _rcb)
    _r['clip']['hi'].on_change('value', _rcb)
    _r['clip']['steps'].on_change('value', _rcb)


# --- Shared widget callbacks ---
for _name, *_ in SLIDER_DEFS:
    sliders[_name]['slider'].on_change('value_throttled', on_shared_change)

w_row_agg.on_change('value', on_shared_change)
w_func.on_change('value', on_shared_change)
w_folder.on_change('value', on_folder_change)
w_multichoice.on_change('value', on_multichoice_change)
w_plot_stat.on_change('active', on_plot_stat_change)

for _name, *_ in SLIDER_DEFS:
    def _make_shared_range_cb(name):
        sg = sliders[name]
        def cb(attr, old, new):
            _update_slider_range(sg)
            refresh_all()
        return cb
    _cb = _make_shared_range_cb(_name)
    sliders[_name]['lo'].on_change('value', _cb)
    sliders[_name]['hi'].on_change('value', _cb)
    sliders[_name]['steps'].on_change('value', _cb)


# --- Layout ---
subtraction_children = [w_row_agg, w_func]
for _name, *_ in SLIDER_DEFS:
    sg = sliders[_name]
    subtraction_children.append(sg['slider'])
    subtraction_children.append(row(sg['lo'], sg['hi'], sg['steps']))
subtraction_col.children = subtraction_children

controls = column(
    Div(text="<b>Image</b>"),
    w_folder,
    w_multichoice,
    w_per_image_toggle,
    per_image_col,
    w_subtraction_toggle,
    subtraction_col,
    w_copy,
    w_err,
)

layout = row(controls, plot_col, sizing_mode='stretch_both')
curdoc().add_root(layout)
curdoc().title = "Subtract Interactive"

# Initial build
rebuild()
