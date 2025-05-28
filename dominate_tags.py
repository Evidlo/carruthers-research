#!/usr/bin/env python3

__all__ = ['plot', 'caption', 'document', 'itemgrid', 'tags', 'util']

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cols = []
    for a in range(3):
        row = []
        for b in range(3):
            fig, ax = plt.subplots()
            ax.imshow(np.random.random((100, 100)))
            fig.tight_layout()
            row.append(caption("hello", plot(fig)))
        cols.append(row)

    with document('helloooo world') as d:
        gridold(cols, flow='column')
        tags.p("end")

    open('/www/domold.html', 'w').write(d.render())

    with document('hello new') as d:
        with itemgrid(3, flow='column'):
            for a in range(3):
                for b in range(3):
                    fig, ax = plt.subplots()
                    ax.imshow(np.random.random((100, 100)))
                    fig.tight_layout()
                    caption("hello", plot(fig))

    open('/www/dom.html', 'w').write(d.render())