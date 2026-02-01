#!/usr/bin/env python3
# Interactive Bokeh app for image subtraction and column analysis
# Run with: bokeh serve subtract_interactive.py --show --allow-websocket-origin=*

import json
import math
import numpy as np
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import (ColumnDataSource, Slider, Spinner, Select,
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


p = dict(
    file=_get('file', 'oob_wfi'),
    clip=_get('clip', 25.0),
    clip_lo=_get('clip_lo', 0.0),
    clip_hi=_get('clip_hi', 5000.0),
    clip_steps=_get('clip_steps', 500),
    func=_get('func', 'x = x + np.mean(x, axis=1, keepdims=True) / a'),
    groups=_get('groups', json.dumps(DEFAULT_GROUPS)),
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

if p['file'] not in FILE_OPTIONS:
    p['file'] = 'oob_wfi'
try:
    json.loads(p['groups'])
except (json.JSONDecodeError, TypeError):
    p['groups'] = json.dumps(DEFAULT_GROUPS)


def _step(lo, hi, steps):
    return (hi - lo) / max(steps, 1)


def _format(step):
    if step <= 0 or step >= 1:
        return "0[.]0"
    d = min(math.ceil(-math.log10(step)), 6)
    return "0." + "0" * d


# --- Computation ---
def compute(file_name, clip_level, func_str, a, b, c, col_groups):
    x = np.load(f'images/{file_name}.npy') / 300
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


# --- Initial computation ---
col_groups_init = json.loads(p['groups'])
squish, img = compute(
    p['file'], p['clip'], p['func'],
    p['a'], p['b'], p['c'], col_groups_init)
nrows, ncols = img.shape

# --- Data Sources ---
line_source = ColumnDataSource(make_line_data(squish, col_groups_init))
img_source = ColumnDataSource(dict(image=[img[::-1]], dw=[ncols], dh=[nrows]))
vline_source = ColumnDataSource(make_vline_data(ncols, nrows, col_groups_init))

# --- Figures ---
fig_lines = figure(width=675, height=525, title="Column Groups",
                   tools="pan,wheel_zoom,box_zoom,reset,save")
fig_lines.multi_line(xs='xs', ys='ys', source=line_source,
                     line_color='colors', line_width=1.5,
                     legend_field='labels')
fig_lines.legend.click_policy = "hide"

mapper = LinearColorMapper(palette=Viridis256, low=0, high=p['clip'])
fig_img = figure(width=675, height=525, title="Image",
                 x_range=DataRange1d(range_padding=0),
                 y_range=DataRange1d(range_padding=0),
                 match_aspect=True,
                 tools="pan,wheel_zoom,box_zoom,reset,save")
fig_img.image(image='image', source=img_source,
              x=0, y=0, dw='dw', dh='dh', color_mapper=mapper)
fig_img.multi_line(xs='xs', ys='ys', source=vline_source,
                   line_color='red', line_width=2)

# --- Widgets ---
w_file = Select(title="Image", value=p['file'],
                options=FILE_OPTIONS, width=200)

_clip_s = _step(p['clip_lo'], p['clip_hi'], p['clip_steps'])
w_clip = Slider(title="Clip Level", value=p['clip'],
                start=p['clip_lo'], end=p['clip_hi'],
                step=_clip_s, format=_format(_clip_s), width=250)
w_clip_lo = Spinner(title="Start", value=p['clip_lo'], step=1, width=80)
w_clip_hi = Spinner(title="Stop", value=p['clip_hi'], step=1, width=80)
w_clip_steps = Spinner(title="Steps", value=p['clip_steps'], step=1, low=1, width=80)

w_func = TextInput(title="Function", value=p['func'], width=300)
w_groups = TextInput(title="Column Groups", value=p['groups'], width=300)

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


# --- Copy Settings button (client-side JS) ---
_url_args = dict(
    w_file=w_file, w_clip=w_clip,
    w_clip_lo=w_clip_lo, w_clip_hi=w_clip_hi, w_clip_steps=w_clip_steps,
    w_func=w_func, w_groups=w_groups,
    w_a=w_a, w_a_lo=w_a_lo, w_a_hi=w_a_hi, w_a_steps=w_a_steps,
    w_b=w_b, w_b_lo=w_b_lo, w_b_hi=w_b_hi, w_b_steps=w_b_steps,
    w_c=w_c, w_c_lo=w_c_lo, w_c_hi=w_c_hi, w_c_steps=w_c_steps,
)

w_copy = Button(label="Copy Settings", width=150)
w_copy.js_on_click(CustomJS(args=_url_args, code="""
    const m = {
        file: w_file, clip: w_clip,
        clip_lo: w_clip_lo, clip_hi: w_clip_hi, clip_steps: w_clip_steps,
        func: w_func, groups: w_groups,
        a: w_a, a_lo: w_a_lo, a_hi: w_a_hi, a_steps: w_a_steps,
        b: w_b, b_lo: w_b_lo, b_hi: w_b_hi, b_steps: w_b_steps,
        c: w_c, c_lo: w_c_lo, c_hi: w_c_hi, c_steps: w_c_steps,
    };
    const p = new URLSearchParams();
    for (const [k, w] of Object.entries(m)) {
        p.set(k, String(w.value));
    }
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
    setTimeout(() => { cb_obj.label = "Copy Settings"; }, 1500);
"""))


# --- Server callbacks ---
def refresh():
    try:
        col_groups = json.loads(w_groups.value)
        s, im = compute(
            w_file.value, w_clip.value, w_func.value,
            w_a.value, w_b.value, w_c.value, col_groups)
        nr, nc = im.shape

        line_source.data = make_line_data(s, col_groups)
        img_source.data = dict(image=[im[::-1]], dw=[nc], dh=[nr])
        vline_source.data = make_vline_data(nc, nr, col_groups)

        mapper.high = max(w_clip.value, 1e-6)
        w_err.text = ""
    except Exception as e:
        w_err.text = f'<span style="color: red">{e}</span>'


def on_change(attr, old, new):
    refresh()


def make_range_cb(slider, lo, hi, steps):
    def cb(attr, old, new):
        slider.start = lo.value
        slider.end = hi.value
        step = (hi.value - lo.value) / max(steps.value, 1)
        slider.step = step
        slider.format = _format(step)
        refresh()
    return cb


# Sliders: update on mouse release
for s in [w_clip, w_a, w_b, w_c]:
    s.on_change('value_throttled', on_change)

# Dropdown and text inputs
w_file.on_change('value', on_change)
w_func.on_change('value', on_change)
w_groups.on_change('value', on_change)

# Spinners: update slider range/step and recompute
for slider, lo, hi, steps in [(w_clip, w_clip_lo, w_clip_hi, w_clip_steps),
                               (w_a, w_a_lo, w_a_hi, w_a_steps),
                               (w_b, w_b_lo, w_b_hi, w_b_steps),
                               (w_c, w_c_lo, w_c_hi, w_c_steps)]:
    cb = make_range_cb(slider, lo, hi, steps)
    lo.on_change('value', cb)
    hi.on_change('value', cb)
    steps.on_change('value', cb)


# --- Layout ---
controls = column(
    Div(text="<b>Image</b>"),
    w_file, w_groups,
    Div(text="<b>Clipping</b>"),
    w_clip, row(w_clip_lo, w_clip_hi, w_clip_steps),
    Div(text="<b>Subtraction</b>"),
    w_func,
    w_a, row(w_a_lo, w_a_hi, w_a_steps),
    w_b, row(w_b_lo, w_b_hi, w_b_steps),
    w_c, row(w_c_lo, w_c_hi, w_c_steps),
    w_copy,
    w_err,
)

layout = row(controls, fig_lines, fig_img)
curdoc().add_root(layout)
curdoc().title = "Subtract Interactive"
