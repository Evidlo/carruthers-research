#!/usr/bin/env python3

__all__ = ['plot', 'caption', 'document', 'itemgrid', 'tags', 'util', 'slider']

from dominate import tags, document, util
from io import BytesIO
import imageio
import base64
import tempfile

import matplotlib
import matplotlib.animation
import numpy as np

def plot(content, title=None, format=None, matkwargs={}, **kwargs):
    """Create HTML plot from Matplotlib figure/anim

    Args:
        content (Figure, Animation, or str): Generate image or animation
            if given a matplotlib Figure or Animation.  Use `content` as
            <img> src if given str
        format (str): format to use when saving matplotlib Figure/Animation
        matkwargs: extra matplotlib arguments
        **kwargs: extra dominate arguments

    Returns:
        dominate.tags.img
        or dominate.tags.figure if `title` is given
    """

    # handle artists
    content = content.figure if hasattr(content, 'figure') else content

    # if given path to image
    if isinstance(content, str):
        src = content

    elif isinstance(content, matplotlib.figure.Figure):
        buff = BytesIO()
        format = 'png' if format is None else format
        with np.errstate(under='ignore'):
            content.savefig(buff, format=format, **matkwargs)
        src = 'data:image/{};base64,{}'.format(
            format,
            base64.b64encode(buff.getvalue()).decode()
        )

    elif isinstance(content, matplotlib.animation.Animation):
        # save animation to temporary file and load bytes
        format = 'gif' if format is None else format
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=True) as tmpfile:
            anim.save(tmpfile.name, **matkwargs)
            tmpfile.seek(0)
            src = 'data:image/{};base64,{}'.format(
                format,
                base64.b64encode(tmpfile.read()).decode()
            )

    elif content is None:
        src = ''

    else:
        raise TypeError(f"Unsupported object {type(content)}")

    return tags.img(src=src, **kwargs)


def caption(title, *args, flow='row', **kwargs):
    """Wraps a set of elements in a <figure> w/ <figcaption>

    Args:
        title (str): title to put in figcaption
        *args (list[...]): list of items to put in figure
    """
    kwargs['style'] = f"""
    display: inline-flex;
    flex-direction: {flow};
    border: 1px solid black;
    """ + kwargs.get('style', "")
    return tags.figure(
        tags.figcaption(title),
        tags.div(*args, **kwargs),
        style="margin:5pt;"
    )


class itemgrid(tags.div):
    """Create a CSS grid of items

    Args:
        length (int): number of items per col/row
        *args: arguments to pass to underlying `dominate.div` tag
        flow (str): flow items as either 'row' or 'col' first
        **kwargs: kwargs to pass to underyling `dominate.div` tag
    """

    def __init__(self, length, *args, flow='row', **kwargs):

        if flow == 'row':
            grid_template = f'grid-template-columns: {"min-content " * length}'
        else:
            grid_template = f'grid-template-rows: {"min-content " * length}'

        kwargs['style'] = f"""
        display: grid;
        {grid_template};
        grid-auto-flow: {flow};
        """ + kwargs.get('style', "")

        super().__init__(*args, **kwargs)

SLIDER_SCRIPT = r'''
// make scope of entire script private
(() => {{
// --- get HTML elements ---
var labels = {labels}
// get elements relative to this script tag
var s = document.currentScript;
// get overall container and elements
var c = s.parentNode.parentNode;
var slider = c.querySelector("#slider")
var counter = c.querySelector("#counter")
var playpause = c.querySelector("#playpause")
// slider/script div is last element.  drop it to get all others
var items = Array.from(c.children).slice(0, -1)

// hide all items
for ([index, item] of items.entries()) {{
    item.style.display = 'none';
}}

// --- slider input ---
var current = items[slider.valueAsNumber];
// hide previous element and show current element when slider changes
slider.oninput = function() {{
    // make sure to load next image before hiding current to prevent flashing
    next = items[this.valueAsNumber];
    next.decode().then(() => {{
        counter.innerHTML = labels[slider.valueAsNumber]
        if (current) {{
            current.style.display = 'none';
        }}
        current = next;
        current.style.display = 'unset';
    }})
}}
slider.oninput()

// --- autoplay button ---
function increment() {{
    slider.value = (slider.valueAsNumber + 1) % (parseInt(slider.max) + 1)
    slider.oninput()
}}
var playing = false
var timer = null
playpause.onclick = function() {{
    if (playing) {{
        window.clearInterval(timer)
    }} else {{
        timer = window.setInterval(increment, 300)
    }}
    playing = !playing
}}
}})();
'''

def slider(*args, labels=None, **kwargs):
    if labels is None:
        labels = [f"{n}" for n in range(len(args))]
    elif type(labels) is str:
        labels = [f"{labels} {n}" for n in range(len(args))]
    return tags.div(
        *args,
        tags.div(
            tags.label(id="counter", _for="slider"),
            tags.input_(id="slider", name="slider", type="range", max=len(args)-1, value="0"),
            tags.button("⏯", id="playpause"),
            tags.script(util.raw(SLIDER_SCRIPT.format(labels=labels)), defer=True),
            style="display: flex; align-items: center; justify-content: center"
        ),
    )



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    with document('hello new') as d:
        with itemgrid(3, flow='column'):
            for a in range(3):
                for b in range(3):
                    fig, ax = plt.subplots()
                    ax.imshow(np.random.random((50, 50)))
                    fig.tight_layout()
                    caption("hello", plot(fig))
    open('/www/dom.html', 'w').write(d.render())

    with document('hello') as d:
        with caption('testing'):
            slider(*[plot(plt.imshow(x)) for x in np.random.random((20, 50, 50))])

    open('/www/dom2.html', 'w').write(d.render())
