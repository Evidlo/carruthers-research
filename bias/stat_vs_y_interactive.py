#!/usr/bin/env python3
# Interactive Bokeh app for stat vs y analysis
# Run with: bokeh serve stat_vs_y_interactive.py --show --allow-websocket-origin=*

import json
import numpy as np
from pathlib import Path
import traceback
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import (ColumnDataSource, Slider, Select,
                          TextInput, Div, LinearColorMapper, DataRange1d,
                          Range1d, CustomJS, Button, Span, TapTool, LinearAxis)
from bokeh.events import Tap
from bokeh.layouts import row, column
from bokeh.palettes import Viridis256

FILE_OPTIONS = [f.stem for f in sorted(Path('images').glob('*.npy'))]

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


# --- Load images once ---
images = {f: np.load(f'images/{f}.npy') / 300 for f in FILE_OPTIONS}


# --- Computation ---
def compute(file_name, clip_level, selected_col, row_stat_str, clip_rows):
    x = images[file_name].copy()
    h, w = x.shape
    if clip_rows is not None:
        x = x[clip_rows[0]:clip_rows[1]]
    nrows, ncols = x.shape

    # Compute row statistic via exec
    ns = {'x': x, 'np': np}
    exec(row_stat_str, ns)
    stat = ns.get('s')
    if stat is not None:
        stat = stat.flatten()
    else:
        stat = np.sum(x, axis=1)

    # Selected column data
    sel_col_idx = min(max(0, selected_col), ncols - 1)
    selected_data = x[:, sel_col_idx]

    clipped_x = np.clip(x, 0, clip_level)

    # Compute percentiles for scatter plot limits
    stat_pos, x_pos = stat, x
    stat_start, stat_stop = (np.percentile(stat_pos, [10, 90]) if len(stat_pos) > 0
                          else (1, 10))
    y_start, y_stop = (np.percentile(x_pos, [5, 95]) if len(x_pos) > 0
                    else (1, 10))

    return (
        clipped_x, stat, selected_data,
        (stat_start, stat_stop, y_start, y_stop)
    )


def make_scatter_data(stat, selected_data):
    # Filter out non-positive values for log scale
    mask = (stat > 0) & (selected_data > 0)
    # return dict(x=stat[mask].tolist(), y=selected_data[mask].tolist())
    return dict(x=stat.tolist(), y=selected_data.tolist())


# --- Widgets ---
_init_file_str = _get('file', 'oob_nfi_l0')
if _init_file_str not in FILE_OPTIONS and FILE_OPTIONS:
    _init_file_str = FILE_OPTIONS[0]

w_file = Select(title="Image", options=FILE_OPTIONS, value=_init_file_str, width=300)
w_row_stat = TextInput(title="Row Statistic",
                       value=_get('row_stat', 's = np.sum(x, axis=1)'),
                       width=300)
w_clip_rows = TextInput(title="Clip Rows",
                        value=_get('clip_rows', 'None'),
                        width=300)
w_clip = Slider(title="Image Clip Level", value=_get('clip', 50.0),
                start=1, end=500, step=1, width=250)
w_selected_col = Slider(title="Selected Column", value=_get('selected_col', 300),
                        start=0, end=1000, step=1, width=250)
w_err = Div(text="", width=300)

# --- Data sources ---
selected_y_src = ColumnDataSource(dict(x=[], y=[]))
stat_line_src = ColumnDataSource(dict(x=[], y=[]))
img_src = ColumnDataSource(dict(image=[], dw=[], dh=[]))
scatter_src = ColumnDataSource(dict(x=[], y=[]))

mapper = LinearColorMapper(palette=Viridis256, low=0, high=50)

# --- Figures ---
# Plot 1: Line plot (selected column + row statistic on second y-axis)
fig_lines = figure(title="Row Profiles",
                   x_axis_label="Row", y_axis_label="Selected Y",
                   y_range=Range1d(),
                   sizing_mode='stretch_both', min_height=250,
                   tools="pan,wheel_zoom,box_zoom,reset,save")
fig_lines.yaxis.axis_label_text_color = 'blue'

# Selected Y line (left y-axis)
fig_lines.line(x='x', y='y', source=selected_y_src,
               line_color='blue', line_width=1.5,
               legend_label='Selected Y')

# Row Statistic line (right y-axis)
fig_lines.extra_y_ranges = {"stat": Range1d()}
fig_lines.add_layout(LinearAxis(y_range_name="stat",
                                axis_label="Row Statistic",
                                axis_label_text_color='red'), 'right')
fig_lines.line(x='x', y='y', source=stat_line_src,
               y_range_name="stat",
               line_color='red', line_width=1.5,
               legend_label='Row Statistic')

fig_lines.legend.click_policy = "hide"

# Plot 2: 2D image with column grouping lines
fig_img = figure(title="Image",
                 x_range=DataRange1d(range_padding=0),
                 y_range=DataRange1d(range_padding=0),
                 match_aspect=True,
                 sizing_mode='scale_height', width=400, height=400,
                 tools="pan,wheel_zoom,box_zoom,reset,save")
fig_img.y_range.flipped = True
fig_img.image(image='image', source=img_src, x=0, y=0, dw='dw', dh='dh',
              color_mapper=mapper)

# Selected column marker (vertical line)
col_span = Span(location=300, dimension='height', line_color='cyan',
                line_width=2, line_dash='dashed')
fig_img.add_layout(col_span)

# Add tap tool for click-to-select column
fig_img.add_tools(TapTool())

# Plot 3: Scatter plot (stat vs selected column)
fig_scatter = figure(title="Statistic vs Y",
                     x_axis_label="Row Statistic", y_axis_label="Selected Y",
                     x_axis_type="log", y_axis_type="linear",
                     x_range=Range1d(1, 10), y_range=Range1d(1, 10),
                     sizing_mode='stretch_both', min_height=250,
                     tools="pan,wheel_zoom,box_zoom,reset,save")
fig_scatter.xaxis.axis_label_text_color = 'red'
fig_scatter.yaxis.axis_label_text_color = 'blue'
fig_scatter.scatter(x='x', y='y', source=scatter_src, size=3, alpha=0.5, color='black')

# --- Copy Settings button ---
w_copy = Button(label="Copy Settings", width=150)
w_copy.js_on_click(CustomJS(args=dict(
    w_file=w_file,
    w_row_stat=w_row_stat,
    w_clip_rows=w_clip_rows,
    w_clip=w_clip,
    w_selected_col=w_selected_col,
), code="""
    const p = new URLSearchParams();
    p.set('file', w_file.value);
    p.set('row_stat', w_row_stat.value);
    p.set('clip_rows', w_clip_rows.value);
    p.set('clip', String(w_clip.value));
    p.set('selected_col', String(w_selected_col.value));

    const url = window.location.origin + window.location.pathname + '?' + p.toString();
    navigator.clipboard.writeText(url);
    cb_obj.label = "Copied!";
    setTimeout(() => { cb_obj.label = "Copy Settings"; }, 1500);
"""))


# --- Server callbacks ---
def refresh(update_scatter_limits=True):
    fname = w_file.value
    if not fname:
        selected_y_src.data = dict(x=[], y=[])
        stat_line_src.data = dict(x=[], y=[])
        img_src.data = dict(image=[], dw=[], dh=[])
        scatter_src.data = dict(x=[], y=[])
        return
    try:
        clip_rows_val = w_clip_rows.value.strip()
        clip_rows = None if clip_rows_val.lower() == 'none' else json.loads(clip_rows_val)
        clip_level = w_clip.value
        selected_col = int(w_selected_col.value)

        im, stat, selected_data, percentiles = compute(
            fname, clip_level, selected_col, w_row_stat.value, clip_rows)
        stat_start, stat_stop, y_start, y_stop = percentiles
        nr, nc = im.shape

        # Update selected col slider range
        w_selected_col.end = nc - 1

        # Update line plot
        row_idx = np.arange(len(selected_data)).tolist()
        selected_y_src.data = dict(x=row_idx, y=selected_data.tolist())
        stat_line_src.data = dict(x=row_idx, y=stat.tolist())

        # Update y-ranges for line plot
        sel_min, sel_max = selected_data.min(), selected_data.max()
        sel_pad = max((sel_max - sel_min) * 0.05, 1e-6)
        fig_lines.y_range.start = sel_min - sel_pad
        fig_lines.y_range.end = sel_max + sel_pad

        stat_min, stat_max = stat.min(), stat.max()
        stat_pad = max((stat_max - stat_min) * 0.05, 1e-6)
        fig_lines.extra_y_ranges['stat'].start = stat_min - stat_pad
        fig_lines.extra_y_ranges['stat'].end = stat_max + stat_pad

        # Update image plot
        mapper.high = max(clip_level, 1e-6)
        img_src.data = dict(image=[im], dw=[nc], dh=[nr])

        # Update column marker
        col_span.location = selected_col

        # Update scatter plot data
        scatter_src.data = make_scatter_data(stat, selected_data)

        # Only update scatter limits when not just changing selected_col
        if update_scatter_limits:
            fig_scatter.x_range.start = stat_start
            fig_scatter.x_range.end = stat_stop
            fig_scatter.y_range.start = y_start
            fig_scatter.y_range.end = y_stop

        w_err.text = ""
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(e))
        print(tb_str)
        w_err.text = f'<span style="color: red">{e}</span>'


def on_change(attr, old, new):
    refresh(update_scatter_limits=True)


def on_col_change(attr, old, new):
    refresh(update_scatter_limits=False)


# Wire up callbacks
w_file.on_change('value', on_change)
w_row_stat.on_change('value', on_change)
w_clip_rows.on_change('value', on_change)
w_clip.on_change('value_throttled', on_change)
w_selected_col.on_change('value_throttled', on_col_change)


def on_tap(event):
    if event.x is not None and event.x >= 0:
        col = int(round(event.x))
        w_selected_col.value = col
        col_span.location = col
        refresh(update_scatter_limits=False)


fig_img.on_event(Tap, on_tap)

# Also update the column span immediately on slider change (client-side)
w_selected_col.js_on_change('value', CustomJS(args=dict(span=col_span), code="""
    span.location = cb_obj.value;
"""))

# --- Layout ---
controls = column(
    w_file,
    w_row_stat,
    w_clip_rows,
    w_clip,
    w_selected_col,
    w_copy,
    w_err,
    width=320,
)

plots = column(
    fig_lines,
    row(fig_img, fig_scatter, sizing_mode='stretch_both'),
    sizing_mode='stretch_both',
)

layout = row(controls, plots, sizing_mode='stretch_both')
curdoc().add_root(layout)
curdoc().title = "Stat vs Y"

# Initial refresh
refresh()
