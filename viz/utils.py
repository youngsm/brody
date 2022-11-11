import matplotlib.colors as mcolors
import numpy as np

from ..misc_utils import get_mask

__all__ = ['hex_to_rgb', 'rgb_to_dec', 'get_continuous_cmap', 'to_gif', 'get_mask']


def to_polar(positions):
    pos = np.atleast_2d(positions)
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    return theta, r


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]]
                    for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def to_gif(input, output, rm=False, show=True, **kwargs):
    '''
    Converts a series of png files to a gif file using ImageMagick.

    input: string of the glob to use
    output: string of the output file name (with extension)
    rm: boolean, whether to remove the png files after conversion
    '''
    kwargs['delay'] = kwargs.get('delay', 100)

    def transform_kwargs(kwargs):
        '''
        >>> transform_kwargs(dict(a=1, b=2))
        ['-a', '1', '-b', '2']
        '''

        call_list = []
        for k in kwargs:
            call_list.extend([''.join(['-', k]), str(kwargs[k])])
        return call_list

    import subprocess
    subprocess.call(['convert', *transform_kwargs(kwargs), input, output])
    if rm:
        subprocess.call(['rm', input])
    if show:
        from IPython.display import Image
        return Image(filename=output)
