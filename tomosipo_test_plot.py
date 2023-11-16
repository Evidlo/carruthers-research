#!/usr/bin/env python3

import numpy as np
import torch as t
from functools import wraps
import tomosipo as ts

def torch2np(args, selected_args, device):
    """Convert a tensor to numpy array"""
    for selected_arg in selected_args:
        if type(args[selected_arg]) is t.Tensor:
            # save device of last detected tensor
            device = args[selected_arg].device.type
            args[selected_arg] = args[selected_arg].detach().cpu().numpy()

    return args, device

def np2torch(item, device):
    """Convert numpy array to tensor"""
    if type(item) is np.ndarray:
        return t.from_numpy(item).to(device=device)
    elif type(item) is t.Tensor:
        return item.to(device)
    elif item is None:
        return
    else:
        return item
        # raise ValueError(f"Unknown item type {type(item)}.  Expected numpy array or pytorch tensor")

def handle_torch(args=None, kwargs=None, ret_args=None):
    """Automatically convert from pytorch tensor to numpy array when function is
    called and convert back when function returns

    If any of the selected input arguments are a tensor, convert them to ndarray
    and convert the selected return values back to tensors. If neither `args`
    nor `kwargs` are given, try to convert any torch Tensor
    we see.

    Args:
        args (list of int): positional arguments to convert
        kwargs (list of str): keyword arguments to convert
        ret_args (list of int): position return values to convert back

    Returns:
        function which handles CUDA tensors

    Usage:
        @handle_torch()
        def my_func(foo):
            ...
            return result_ndarray

        >>> my_func(input_tensor)
        result_tensor

        >>> my_func(input_ndarray)
        result_ndarray

        @handle_torch(args=[0], kwargs=['bar'])
        def my_func2(foo, bar=None):
            ...
            return some_ndarray
    """


    def decorator(func):
        @wraps(func)
        def wrapper(*f_args, **f_kwargs):
            f_args = list(f_args)
            device = None

            # ----- Calling -----

            # if no positional/keyword arguments selected, use all
            if args is None and kwargs is None:
                selected_args = range(len(f_args))
                selected_kwargs = f_kwargs.keys()
            else:
                selected_args = args
                selected_kwargs = kwargs
            # convert requested args from cuda
            if selected_args is not None:
                f_args, device = torch2np(f_args, selected_args, device)

            # convert requested kwargs from cuda
            if selected_kwargs is not None:
                f_kwargs, device = torch2np(f_kwargs, selected_kwargs, device)


            # call the function
            old_return = func(*f_args, **f_kwargs)

            # ----- Returning -----

            # if any of the inputs were pytorch tensors, convert the selected return values to tensors
            if device is not None:
                if ret_args is not None:
                    assert type(old_return) is tuple, "Function {func.__name__} should have returned multiple arguments"
                    new_return = list(old_return)
                    for arg in ret_args:
                        new_return[arg] = np2torch(old_return[arg], device)
                else:
                    if type(old_return) is tuple:
                        new_return = [np2torch(item, device) for item in old_return]
                    else:
                        new_return = np2torch(old_return, device)
            else:
                new_return = old_return

            return new_return

        return wrapper

    return decorator

def rescale_max(x, rescale):
    """Rescale a stack of images by the global max or max in each image

    Args:
        x (torch.Tensor or numpy.ndarray): input data of shape (num_images, width, height)
            or (num_images, width, height, 3) for RGB images
        rescale (str): if 'frame', rescale max of each image to 255.  If 'sequence',
            rescale max of whole image sequence to 255. (default 'frame')
    """
    axes = tuple(range(len(x.shape)))

    x[x < np.finfo(float).eps] = 0

    # handle intensity scaling
    if rescale == 'frame':
        # scale each frame separately
        # x *= np.expand_dims(255 / np.max(x, axes[1:]), axes[1:])
        maxx = np.max(x, axes[1:])
        x *= np.expand_dims(np.divide(255, maxx, where=(maxx != 0)), axes[1:])
    elif rescale == 'sequence':
        # x *= 255 / np.max(x)
        maxx = np.max(x)
        x *= np.divide(255, maxx, where=(maxx != 0))

    return x


@handle_torch()
def save_gif(savefile, x, rescale='frame', duration=100):
    """Save image sequence as gif

    Args:
        savefile (str): path to save location
        x (torch.Tensor or numpy.ndarray): input data of shape (num_images, width, height)
            or (num_images, width, height, 3) for RGB images
        rescale (str): if 'frame', rescale max of each image to 255.  If 'sequence',
            rescale max of whole image sequence to 255. (default 'frame')
        duration (int): delay in ms between frames
    """

    from PIL import Image

    x = rescale_max(x, rescale)

    # if grayscale input, convert to RGB
    if len(x.shape) == 3:
        imgs = [Image.fromarray(img.astype(np.uint8), mode='L') for img in x]
    elif len(x.shape) == 4:
        imgs = [Image.fromarray(img.astype(np.uint8), mode='RGB') for img in x]
    else:
        raise ValueError('Invalid size for x')

    # duration is the number of milliseconds between frames
    imgs[0].save(savefile, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


def preview3d(x, volume=None, size=300, fov=45, num_images=20):
    """Create a rotating animated preview of a density

    Args:
        x (ndarray): volume to preview of shape (width, height, depth) or
            (width, height, depth, num_channels) for multi-channel measurement
        volume (tomosipo VolumeGeometry): optional volume geometry if grid is not regular cubes
        size (int): width of each square preview image
        fov (float): camera FOV for generating preview, in degrees
        num_images (int): number of images in preview

    Returns:
        images (ndarray): stack of images of dimension (num_images, size, size) or
            (num_images, size, size, num_channels)
    """

    if volume is None:
        volume = ts.volume(pos=0, size=1, shape=x.shape[:3])
    projections = []
    for theta in np.linspace(0, 2 * np.pi, num_images):
        projections.append(conebeam_proj(
            (size, size),
            max(volume.size) * np.array((2 * np.cos(theta), 2 * np.sin(theta), 1)),
            (fov, fov)
        ))
    projections = ts.concatenate(projections)
    op = ts.operator(volume, projections)

    # if multiple channels, process each separately
    if len(x.shape) == 4:
        results = []
        for chan in np.rollaxis(x, 3):
            results.append(op(chan))
        result = np.stack(results, axis=-1)
    else:
        result = op(x)

    # ts.operator returns array of shape (width, num_images, height).  fix this
    if type(result) is np.ndarray:
        return np.moveaxis(result, (0, 1, 2), (1, 0, 2))
    else:
        return result.moveaxis((0, 1, 2), (1, 0, 2))


def conebeam_proj(det_shape, det_pos, fov, det_orientation=None):
    """Create tomosipo cone beam projector

    Tomosipo conebeam projection is meant for MRI/CT which has a point source
    with rays that fan out to a detector on the opposite side of the object.
    But GLIDE is the opposite, with the spacecraft located at the intersection
    point of the rays.  We treat the tomosipo "source" as the detector
    and the tomosipo "detector" is where the line integrals stop.

    Provided coordinates should be (X, Y, Z).
    Returned object follows Tomosipo coordinate convention, (Z, Y, X).

    Args:
        det_shape (tuple of int): size of detector in pixels (width, height)
        det_pos (tuple of float): position of detector in GSE coordinates (X, Y, Z) (Earth radii)
        det_orientation (ndarray or None): array of size (3, 3) containing camera
            If None, point at origin
        fov (tuple of float): FOV in degrees for u and v directions

    Returns:
        tomosipo.geometry.ConeVectorGeometry
    """
    import math

    if det_orientation is None:
        # choose default u, v so spacecraft is aligned with ecliptic plane and looking at origin
        det_u = np.cross(np.array(det_pos), (0, 0, 1))
        det_v = np.cross(np.array(det_pos), det_u)
        det_w = np.cross(det_u, det_v)
    else:
        det_u = det_orientation[:, 2]
        det_v = det_orientation[:, 0]
        det_w = -det_orientation[:, 1]

    # normalize u, v, w
    det_u = det_u / np.linalg.norm(det_u)
    det_v = det_v / np.linalg.norm(det_v)
    det_w = det_w / np.linalg.norm(det_w)

    # project detector position onto perpendicular plane
    proj = det_pos + (np.dot(det_pos, -det_w) / np.linalg.norm(det_w)**2) * det_w
    # find point opposite of detector
    endplane_pos = det_pos - (det_pos - proj) * 2
    distance = np.linalg.norm(det_pos - endplane_pos)
    endplane_size = (
        distance * 2 * math.tan(math.radians(fov[0] / 2)),
        distance * 2 * math.tan(math.radians(fov[1] / 2))
    )
    # scale u, v for virtual detector
    det_u *= endplane_size[0] / det_shape[0]
    det_v *= endplane_size[1] / det_shape[1]

    return ts.cone_vec(
        shape=det_shape,
        det_pos=endplane_pos,
        src_pos=det_pos,
        # flip the tomosipo detector around so it's looking in the same direction
        # as the source
        det_u=det_u,
        det_v=det_v
    )
