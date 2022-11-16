from typing import List

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .utils import set_custom_style

__exports__ = ["hist", "hist2d", "hist3d"]

@set_custom_style
def hist(
        data: List[h5py.Dataset] or h5py.Dataset, bins=None, xlabel=None, ylabel=None,
        title=None, log=False, error=True, grid=False, legend=True, color=None, norm=None,
        capsize=0, xlog=False, density=False, figsize=None, dpi=None, **kwargs):
    """hist
    
    A thin wrapper around plt.hist that has some sensible defaults.

    Parameters
    ----------
    data : List[h5py.Dataset] or h5py.Dataset
        The data to plot out. Can be a list of datasets or a single dataset.
    bins : List[int or ArrayLike], optional
        Binning for each dataset, in the same format as bins in plt.hist(), 
        by default 200 bins +/- 4 sigma from the mean of each dataset.
    xlabel : str, optional
        Label on the x-axis, by default None
    ylabel : str, optional
        Label on the y-axis, by default None
    title : str, optional
        Title, by default None
    log : bool, optional
        Set the scaling on the y-axis to log, by default False
    error : bool, optional
        Whether to plot errors (sqrt(N)) on each bin, by default True
    grid : bool, optional
        Whether to call plt.grid(), by default False
    legend : bool or str or List[str], optional
        if `True` or `False`, determines whether to plot a legend. If `True`, this
        uses the attrs['description'] of each dataset. If a list of strings,
        it will use the list as legend labels, by default True
    color : List[str] or str, optional
        Colors to use for each dataset being plotted, by default None
    norm : List[float] or float, optional
        A norm to divide histogram counts by, by default None. For example, if you
        ran a 5 MeV simulation and want to find the number of counts per 1 MeV,
        you would set norm=5.
    capsize : int, optional
        Set the capsize of errorbars, by default 0
    xlog : bool, optional
        Set the scaling on the x-axis to log, by default False
    density : bool, optional
        If True, draw and return a probability density: each bin will display the bin's
        raw count divided by the total number of counts and the bin width
        (density = counts / (sum(counts) * np.diff(bins))), so that the area under the
        histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1), by default False
    figsize : tuple, optional
        Set the figure size of the plot, by default plt.rcParams['figure.figsize']
    dpi : float, optional
        Set the dpi of the plot, by default plt.rcParams['figure.dpi']
    """
    fig_kwargs = {}
    if figsize:
        fig_kwargs['figsize'] = figsize
    if dpi:
        fig_kwargs['dpi'] = dpi
    if fig_kwargs:
        plt.figure(**fig_kwargs)

    if isinstance(data, h5py.Dataset):
        datas = [data]
    else:
        datas = data
    bins_set = bins is not None

    if norm is not None and not density:
        norms = norm if np.iterable(norm) else [norm]
    else:
        norms = [1]*len(datas)

    if color is not None:
        colors = color if np.iterable(color) else [color]*len(datas)
    else:
        colors = [None]*len(datas)

    if legend is not None:
        labels = legend if np.iterable(legend) else None
        legend = True

    argsort = np.argsort([np.mean(d) for d in datas])[::-1]
    datas = [datas[i] for i in argsort]
    colors = [colors[i] for i in argsort]
    if legend and labels is not None:
        labels = [labels[i] for i in argsort]

    for i, (d, c, n) in enumerate(zip(datas, colors, norms)):

        if legend:
            if labels is not None:
                label = labels[i]
            else:
                label = d.attrs.get("description", "").capitalize()
        else:
            label = ""

        mean = np.mean(d[:])
        var = np.sqrt(np.var(d[:]))
        if bins is None:
            bins = np.linspace(mean-4*var, mean+4*var, 200)
        elif not bins_set:
            diff = bins[1]-bins[0]
            if mean+3*var > bins[-1]:
                bins = np.arange(bins[0], mean+4*var+diff, diff)
            if mean-3*var < bins[0]:
                bins = np.arange(mean-4*var, bins[-1]+diff, diff)
        counts, edges = np.histogram(d[:], bins, density=density)
        counts = counts.astype(float)
        centers = (edges[:-1] + edges[1:]) / 2

        plot_kwargs = dict(
            drawstyle='steps-mid', linewidth=plt.rcParams["axes.linewidth"],
            label=label, color=c)

        if error:
            plt.errorbar(centers, counts/n, yerr=np.sqrt(counts)/n,
                         **plot_kwargs, capsize=capsize, **kwargs)
        else:
            plt.plot(centers, counts/n, **plot_kwargs, **kwargs)

    plt.xlim(bins[0], bins[-1])

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=12)
    if log:
        plt.yscale('log')
    else:
        plt.ylim(0, None)
    if xlog:
        plt.xscale("log")
    if grid:
        plt.grid()
    if legend:
        plt.legend(frameon=False)
    plt.show()


@set_custom_style
def hist2d(data, bins=[100, 100], xlabel=None, ylabel=None, title=None, log=False, cmap="inferno"):
    """hist2d

    Parameters
    ----------
    data : List[h5py.Dataset]
        The data to plot out. Must be a list of two datasets.
    bins : list[int or ArrayLike], optional
        Bins to plot out, to be inputted to np.histgram2d, by default [100, 100]
    xlabel : str, optional
        Label on the x-axis, by default None
    ylabel : str, optional
        Label on the y-axis, by default None
    title : str, optional
        Title, by default None
    log : bool, optional
        Set the scaling on the colormap to logarithmic, by default False
    cmap : str, optional
        Set the colormap to use, by default "inferno"
    """
    if log:
        Scale = LogNorm()
    else:
        Scale = None

    datax, datay = data
    h, x, y = np.histogram2d(datax, datay, bins=bins)
    plt.pcolor(x, y, h.T, norm=Scale, cmap=cmap)
    plt.colorbar(pad=0.01)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=12)

    plt.show()


@set_custom_style
def hist3d(data, xbins, ybins=None, xlabel=None, ylabel=None, zlabel=None, title=None,
           log=False, cmap="inferno", contour=True, **kwargs):
    """Note that the principal data component is first."""
    def log_tick_formatter(val, pos):
        return f"$10^{{{int(val)}}}$"

    data = np.atleast_1d(data)
    xbins = np.asarray(xbins, dtype=float)
    assert len(data) == len(xbins)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if ybins is None:
        max_cts = max(map(np.max, data))
        ybins = np.arange(0, max_cts+1)

    dataz = [
        np.histogram(d, ybins)[0]
        for d in data
    ]

    y = np.array((ybins[1:]+ybins[:-1])/2)
    x = np.array(xbins)
    x, y = np.meshgrid(x, y)
    z = np.array(dataz).T

    if log:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.log10(z)
        z[np.isinf(z)] = min(z[~np.isinf(z)])
        Scale = LogNorm()
        ax.zaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    else:
        Scale = None

    surf = ax.plot_surface(x, y, z, rstride=2, cstride=2,
                           cmap=cmap, edgecolor='none',
                           norm=Scale, **kwargs)
    if contour:
        ax.contour(x, y, z, zdir='z', offset=np.max(z)+10, cmap=cmap, levels=5)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    if title is not None:
        ax.set_title(title, fontsize=12)

    plt.colorbar(surf, shrink=0.5)
    plt.show()
