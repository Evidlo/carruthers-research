#!/usr/bin/env python3
# Interactive Bokeh app for image subtraction and column analysis
# Run with: bokeh serve subtract_interactive.py --show --allow-websocket-origin=*

import json
import math
import numpy as np
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import (ColumnDataSource, Slider, Spinner, CheckboxGroup,
                          TextInput, Div, LinearColorMapper, DataRange1d,
                          CustomJS, Button)
from bokeh.layouts import row, column
from bokeh.palettes import Viridis256, Category10_10

DEFAULT_GROUPS = [[0, 0.25], [0.25, 0.75], [0.75, 1]]
FILE_OPTIONS = ['oob_wfi', 'oob_nfi', 'sci_wfi', 'sci_nfi']


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


# Shared params
p = dict(
    files=_get('files', 'oob_wfi'),
    func=_get('func', 'x = x + np.mean(x, axis=1, keepdims=True) / a'),
    a=_get('a', 80.0),
    a_lo=_get('a_lo', 0.0),
    a_hi=_get('a_hi', 500.0),
    a_steps=_get('a_steps', 500),
    b=_get('b', 1.0),
    b_lo=_get('b_lo', 0.0),
    b_hi=_get('b_hi', 1.0),
    b_steps=_get('b_steps', 500),
    c=_get('c', 1.0),
    c_lo=_get('c_lo', 0.0),
    c_hi=_get('c_hi', 1.0),
    c_steps=_get('c_steps', 500),
)

_init_files = [f.strip() for f in p['files'].split(',')
               if f.strip() in FILE_OPTIONS]
if not _init_files:
    _init_files = ['oob_wfi']
_init_active = [FILE_OPTIONS.index(f) for f in _init_files]


def _step(lo, hi, steps):
    return (hi - lo) / max(steps, 1)


def _format(step):
    if step <= 0 or step >= 1:
        return "0[.]0"
    d = min(math.ceil(-math.log10(step)), 6)
    return "0." + "0" * d


# --- Load images once ---
images = {f: np.load(f'images/{f}.npy') / 300 for f in FILE_OPTIONS}


# --- Computation ---
def compute(file_name, clip_level, func_str, a, b, c, col_groups):
    x = images[file_name].copy()
    ns = {'x': x, 'a': a, 'b': b, 'c': c, 'np': np}
    exec(func_str, ns)
    x = ns['x']

    nrows, ncols = x.shape
    groups = []
    for frac_lo, frac_hi in col_groups:
        c0 = int(round(frac_lo * ncols))
        c1 = int(round(frac_hi * ncols))
        groups.append(np.median(x[:, c0:c1], axis=1))
    squish = np.stack(groups, axis=-1)

    return np.clip(squish, 0, clip_level), np.clip(x, 0, clip_level)


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
        xpos = frac_lo * ncols
        xs.append([xpos, xpos])
        ys.append([0, nrows])
        xpos = frac_hi * ncols
        xs.append([xpos, xpos])
        ys.append([0, nrows])
    return dict(xs=xs, ys=ys)


# --- Per-image state (pre-create for all images) ---
rows = {}
for _fname in FILE_OPTIONS:
    _clip_val = _get(f'clip.{_fname}', 25.0)
    _groups_val = _get(f'groups.{_fname}', json.dumps(DEFAULT_GROUPS))
    _clip_lo_val = _get(f'clip_lo.{_fname}', 0.0)
    _clip_hi_val = _get(f'clip_hi.{_fname}', 5000.0)
    _clip_steps_val = _get(f'clip_steps.{_fname}', 500)

    try:
        json.loads(_groups_val)
    except (json.JSONDecodeError, TypeError):
        _groups_val = json.dumps(DEFAULT_GROUPS)

    _cs = _step(_clip_lo_val, _clip_hi_val, _clip_steps_val)
    _w_g = TextInput(value=_groups_val, width=300)
    _w_cl = Slider(title="Clip", value=_clip_val,
                   start=_clip_lo_val, end=_clip_hi_val,
                   step=_cs, format=_format(_cs), width=250)
    _w_cl_lo = Spinner(title="Start", value=_clip_lo_val, step=1, width=80)
    _w_cl_hi = Spinner(title="Stop", value=_clip_hi_val, step=1, width=80)
    _w_cl_st = Spinner(title="Steps", value=_clip_steps_val,
                       step=1, low=1, width=80)

    _mp = LinearColorMapper(palette=Viridis256, low=0, high=_clip_val)
    _ls = ColumnDataSource(dict(xs=[], ys=[], colors=[], labels=[]))
    _is = ColumnDataSource(dict(image=[], dw=[], dh=[]))
    _vs = ColumnDataSource(dict(xs=[], ys=[]))

    _fl = figure(title=f"{_fname} — Column Groups",
                 sizing_mode='stretch_both', min_height=200,
                 tools="pan,wheel_zoom,box_zoom,reset,save")
    _fl.multi_line(xs='xs', ys='ys', source=_ls,
                   line_color='colors', line_width=1.5,
                   legend_field='labels')
    _fl.legend.click_policy = "hide"

    _fi = figure(title=_fname,
                 x_range=DataRange1d(range_padding=0),
                 y_range=DataRange1d(range_padding=0),
                 match_aspect=True,
                 sizing_mode='scale_height', width=400, height=400,
                 tools="pan,wheel_zoom,box_zoom,reset,save")
    _fi.image(image='image', source=_is, x=0, y=0, dw='dw', dh='dh',
              color_mapper=_mp)
    _fi.multi_line(xs='xs', ys='ys', source=_vs,
                   line_color='red', line_width=2)

    rows[_fname] = dict(
        w_groups=_w_g, w_clip=_w_cl,
        w_clip_lo=_w_cl_lo, w_clip_hi=_w_cl_hi, w_clip_steps=_w_cl_st,
        line_source=_ls, img_source=_is, vline_source=_vs,
        fig_lines=_fl, fig_img=_fi, mapper=_mp,
    )


# --- Shared Widgets ---
w_checkbox = CheckboxGroup(labels=FILE_OPTIONS, active=_init_active)

w_func = TextInput(title="Function", value=p['func'], width=300)

_a_s = _step(p['a_lo'], p['a_hi'], p['a_steps'])
w_a = Slider(title="a", value=p['a'],
             start=p['a_lo'], end=p['a_hi'],
             step=_a_s, format=_format(_a_s), width=250)
w_a_lo = Spinner(title="Start", value=p['a_lo'], step=1, width=80)
w_a_hi = Spinner(title="Stop", value=p['a_hi'], step=1, width=80)
w_a_steps = Spinner(title="Steps", value=p['a_steps'], step=1, low=1, width=80)

_b_s = _step(p['b_lo'], p['b_hi'], p['b_steps'])
w_b = Slider(title="b", value=p['b'],
             start=p['b_lo'], end=p['b_hi'],
             step=_b_s, format=_format(_b_s), width=250)
w_b_lo = Spinner(title="Start", value=p['b_lo'], step=0.01, width=80)
w_b_hi = Spinner(title="Stop", value=p['b_hi'], step=0.01, width=80)
w_b_steps = Spinner(title="Steps", value=p['b_steps'], step=1, low=1, width=80)

_c_s = _step(p['c_lo'], p['c_hi'], p['c_steps'])
w_c = Slider(title="c", value=p['c'],
             start=p['c_lo'], end=p['c_hi'],
             step=_c_s, format=_format(_c_s), width=250)
w_c_lo = Spinner(title="Start", value=p['c_lo'], step=0.01, width=80)
w_c_hi = Spinner(title="Stop", value=p['c_hi'], step=0.01, width=80)
w_c_steps = Spinner(title="Steps", value=p['c_steps'], step=1, low=1, width=80)

w_err = Div(text="", width=300)


# --- Dynamic layout containers ---
per_image_col = column(visible=False)
plot_col = column(sizing_mode='stretch_both')

w_per_image_toggle = Button(label="\u25b6 Image Controls", width=300)
w_per_image_toggle.js_on_click(CustomJS(args=dict(col=per_image_col), code="""
    col.visible = !col.visible;
    cb_obj.label = col.visible ? "\u25bc Image Controls" : "\u25b6 Image Controls";
"""))


# --- Copy Settings button (client-side JS) ---
_url_args = dict(
    w_checkbox=w_checkbox,
    w_func=w_func,
    w_a=w_a, w_a_lo=w_a_lo, w_a_hi=w_a_hi, w_a_steps=w_a_steps,
    w_b=w_b, w_b_lo=w_b_lo, w_b_hi=w_b_hi, w_b_steps=w_b_steps,
    w_c=w_c, w_c_lo=w_c_lo, w_c_hi=w_c_hi, w_c_steps=w_c_steps,
)
for _fname in FILE_OPTIONS:
    _r = rows[_fname]
    _url_args[f'g_{_fname}'] = _r['w_groups']
    _url_args[f'cl_{_fname}'] = _r['w_clip']
    _url_args[f'cllo_{_fname}'] = _r['w_clip_lo']
    _url_args[f'clhi_{_fname}'] = _r['w_clip_hi']
    _url_args[f'clst_{_fname}'] = _r['w_clip_steps']

_pi_entries = []
for _fname in FILE_OPTIONS:
    _pi_entries.append(
        f"'{_fname}': {{g: g_{_fname}, cl: cl_{_fname}, "
        f"cllo: cllo_{_fname}, clhi: clhi_{_fname}, clst: clst_{_fname}}}")
_pi_js = '{\n        ' + ',\n        '.join(_pi_entries) + '\n    }'

w_copy = Button(label="Copy Settings", width=150)
w_copy.js_on_click(CustomJS(args=_url_args, code=f"""
    const files = {json.dumps(FILE_OPTIONS)};
    const active = w_checkbox.active;
    const pi = {_pi_js};
    const p = new URLSearchParams();

    p.set('files', active.map(i => files[i]).join(','));
    p.set('func', String(w_func.value));
    p.set('a', String(w_a.value));
    p.set('a_lo', String(w_a_lo.value));
    p.set('a_hi', String(w_a_hi.value));
    p.set('a_steps', String(w_a_steps.value));
    p.set('b', String(w_b.value));
    p.set('b_lo', String(w_b_lo.value));
    p.set('b_hi', String(w_b_hi.value));
    p.set('b_steps', String(w_b_steps.value));
    p.set('c', String(w_c.value));
    p.set('c_lo', String(w_c_lo.value));
    p.set('c_hi', String(w_c_hi.value));
    p.set('c_steps', String(w_c_steps.value));

    for (const fname of files) {{
        const r = pi[fname];
        p.set('groups.' + fname, String(r.g.value));
        p.set('clip.' + fname, String(r.cl.value));
        p.set('clip_lo.' + fname, String(r.cllo.value));
        p.set('clip_hi.' + fname, String(r.clhi.value));
        p.set('clip_steps.' + fname, String(r.clst.value));
    }}

    const url = window.location.origin + window.location.pathname + '?' + p.toString();
    const ta = document.createElement('textarea');
    ta.value = url;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    cb_obj.label = "Copied!";
    setTimeout(() => {{ cb_obj.label = "Copy Settings"; }}, 1500);
"""))


# --- Server callbacks ---
def refresh_image(fname):
    r = rows[fname]
    try:
        col_groups = json.loads(r['w_groups'].value)
        s, im = compute(fname, r['w_clip'].value, w_func.value,
                        w_a.value, w_b.value, w_c.value, col_groups)
        nr, nc = im.shape
        line_data = make_line_data(s, col_groups)
        r['line_source'].data = line_data
        r['img_source'].data = dict(image=[im[::-1]], dw=[nc], dh=[nr])
        r['vline_source'].data = make_vline_data(nc, nr, col_groups)
        r['mapper'].high = max(r['w_clip'].value, 1e-6)
        all_ys = [v for sub in line_data['ys'] for v in sub]
        if all_ys:
            lo, hi = min(all_ys), max(all_ys)
            pad = max((hi - lo) * 0.05, 1e-6)
            r['fig_lines'].y_range.start = lo - pad
            r['fig_lines'].y_range.end = hi + pad
        w_err.text = ""
    except Exception as e:
        w_err.text = f'<span style="color: red">{e}</span>'


def refresh_all():
    for i in w_checkbox.active:
        refresh_image(FILE_OPTIONS[i])


def rebuild():
    selected = [FILE_OPTIONS[i] for i in w_checkbox.active]

    # Rebuild per-image controls in the left column
    children = []
    if selected:
        children.append(Div(text="<b>Column Grouping</b>"))
        for fname in selected:
            children.append(rows[fname]['w_groups'])
        for fname in selected:
            r = rows[fname]
            children.extend([
                r['w_clip'],
                row(r['w_clip_lo'], r['w_clip_hi'], r['w_clip_steps']),
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


def on_checkbox_change(attr, old, new):
    rebuild()


# --- Per-image widget callbacks ---
for _fname in FILE_OPTIONS:
    _r = rows[_fname]

    def _make_img_cb(fn):
        def cb(attr, old, new):
            if FILE_OPTIONS.index(fn) in w_checkbox.active:
                refresh_image(fn)
        return cb

    _r['w_groups'].on_change('value', _make_img_cb(_fname))
    _r['w_clip'].on_change('value_throttled', _make_img_cb(_fname))

    def _make_range_cb(fn):
        r = rows[fn]
        def cb(attr, old, new):
            step = _step(r['w_clip_lo'].value, r['w_clip_hi'].value,
                         r['w_clip_steps'].value)
            r['w_clip'].start = r['w_clip_lo'].value
            r['w_clip'].end = r['w_clip_hi'].value
            r['w_clip'].step = step
            r['w_clip'].format = _format(step)
            if FILE_OPTIONS.index(fn) in w_checkbox.active:
                refresh_image(fn)
        return cb

    _rcb = _make_range_cb(_fname)
    _r['w_clip_lo'].on_change('value', _rcb)
    _r['w_clip_hi'].on_change('value', _rcb)
    _r['w_clip_steps'].on_change('value', _rcb)


# --- Shared widget callbacks ---
for _s in [w_a, w_b, w_c]:
    _s.on_change('value_throttled', on_shared_change)

w_func.on_change('value', on_shared_change)
w_checkbox.on_change('active', on_checkbox_change)

for _slider, _lo, _hi, _steps in [(w_a, w_a_lo, w_a_hi, w_a_steps),
                                    (w_b, w_b_lo, w_b_hi, w_b_steps),
                                    (w_c, w_c_lo, w_c_hi, w_c_steps)]:
    def _make_shared_range_cb(sl, lo, hi, st):
        def cb(attr, old, new):
            step = _step(lo.value, hi.value, st.value)
            sl.start = lo.value
            sl.end = hi.value
            sl.step = step
            sl.format = _format(step)
            refresh_all()
        return cb
    _cb = _make_shared_range_cb(_slider, _lo, _hi, _steps)
    _lo.on_change('value', _cb)
    _hi.on_change('value', _cb)
    _steps.on_change('value', _cb)


# --- Layout ---
controls = column(
    Div(text="<b>Image</b>"),
    w_checkbox,
    w_per_image_toggle,
    per_image_col,
    Div(text="<b>Subtraction</b>"),
    w_func,
    w_a, row(w_a_lo, w_a_hi, w_a_steps),
    w_b, row(w_b_lo, w_b_hi, w_b_steps),
    w_c, row(w_c_lo, w_c_hi, w_c_steps),
    w_copy,
    w_err,
)

layout = row(controls, plot_col, sizing_mode='stretch_both')
curdoc().add_root(layout)
curdoc().title = "Subtract Interactive"

# Initial build
rebuild()
