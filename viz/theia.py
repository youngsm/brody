import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chroma.event import BULK_REEMIT, CHERENKOV, SCINTILLATION, SURFACE_DETECT
from geometry.theia import gen_rot
from matplotlib.colors import LogNorm
import h5py
from typing import List
import matplotlib

from .utils import get_mask, get_continuous_cmap, to_polar

__all__ = ["plot_hits", "hist", "hist2d", "hist3d"]

matplotlib.rcParams['figure.figsize'] = [10,7]
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['font.family'] = 'DejaVu Serif'
plt.style.use('dark_background')


def dark_background(fun):
    def wrapper(*args, **kwargs):
        plt.style.use('dark_background')
        res = fun(*args, **kwargs)
        plt.style.use('default')
        return res
    return wrapper


def plot_hits(geo, ev, out, text=f'1 GeV\n$\mu^-$', pmt_type=1, tcut=False, log=False):
    pmt_positions = geo.channel_index_to_position
    pmt_types = geo.channel_index_to_channel_type
    zbottom_l, zbottom_s = pmt_positions[pmt_types == 2][:, 2].min(
    ), pmt_positions[pmt_types == 1][:, 2].min()
    ztop_l, ztop_s = pmt_positions[pmt_types == 2][:, 2].max(
    ), pmt_positions[pmt_types == 1][:, 2].max()

    hit_channels, true_times, charges = ev.channels.hit_channels()
    hit_times = np.random.normal(true_times, 1.0)
    hit_types = pmt_types[hit_channels]
    hit_positions = pmt_positions[hit_channels]
    hit_charges = charges
    flags = ev.channels.flags[ev.channels.hit]

    long_hits = hit_positions[hit_types == 2]
    long_charges = hit_charges[hit_types == 2]
    long_times = true_times[hit_types == 2]
    long_flags = flags[hit_types == 2]
    is_cher_l = get_mask(long_flags, CHERENKOV |
                           SURFACE_DETECT, none_of=BULK_REEMIT).astype(bool)
    is_scint_l = get_mask(long_flags, SCINTILLATION |
                            SURFACE_DETECT, none_of=BULK_REEMIT).astype(bool)
    cher_hits_l = long_hits[is_cher_l]
    cher_times_l = long_times[is_cher_l]
    cher_charges_l = long_charges[is_cher_l]
    other_hits_l = long_hits[~is_cher_l]
    scint_hits_l = long_hits[is_scint_l]
    scint_times_l = long_times[is_scint_l]
    scint_charges_l = long_charges[is_scint_l]

    short_hits = hit_positions[hit_types == 1]
    short_charges = hit_charges[hit_types == 1]
    short_times = true_times[hit_types == 1]
    short_flags = flags[hit_types == 1]
    is_cher_s = get_mask(short_flags, CHERENKOV |
                           SURFACE_DETECT, none_of=BULK_REEMIT).astype(bool)
    is_scint_s = get_mask(short_flags, SCINTILLATION |
                            SURFACE_DETECT, none_of=BULK_REEMIT).astype(bool)
    cher_times_s = short_times[is_cher_s]
    cher_hits_s = short_hits[is_cher_s]
    cher_charges_s = short_charges[is_cher_s]
    other_hits_s = short_hits[~is_cher_s]
    scint_hits_s = short_hits[is_scint_s]
    scint_times_s = short_times[is_scint_s]
    scint_charges_s = short_charges[is_scint_s]

    long_hit_time_res = long_times - \
        np.sqrt(np.sum(
            np.square(long_hits-ev.vertices[0].pos), axis=1))/(300/1.4876301945708197)
    short_hit_time_res = short_times - \
        np.sqrt(np.sum(
            np.square(short_hits-ev.vertices[0].pos), axis=1))/(300/1.5069515555735635)

    fig = plt.figure(figsize=(10, 14), edgecolor='white')
    gridspec = plt.GridSpec(nrows=3, ncols=3, height_ratios=[
        1, 0.8, 1], width_ratios=[0.6, 1, 0.6])

    circ1 = fig.add_subplot(gridspec[1], projection="polar")
    center = fig.add_subplot(gridspec[3:6])
    circ2 = fig.add_subplot(gridspec[7], projection="polar")

    def plot_hits(hit_pos, top, bottom, c=None, s=1, **kwargs):
        top_mask = (hit_pos == top).any(axis=1)
        bottom_mask = (hit_pos == bottom).any(axis=1)
        middle_mask = ~top_mask & ~bottom_mask
        tt, tr = tp = to_polar(hit_pos[top_mask])
        m_theta, m_r = to_polar(hit_pos[middle_mask])
        m_z = hit_pos[middle_mask][:, 2]

        bottom_pts = hit_pos[bottom_mask]
        bottom_pts[:, 0] = -bottom_pts[:, 0]
        bt, br = bp = to_polar(bottom_pts)
        tt -= np.pi/2
        bt -= np.pi/2
        # tt %= 2*np.pi
        # bt %= 2*np.pi

        try:
            ct = c[top_mask]
            cm = c[middle_mask]
            cb = c[bottom_mask]
        except TypeError:
            ct = cm = cb = c

        circ1.scatter(*tp, s=s, c=ct, **kwargs)
        circ2.scatter(*bp, s=s, c=cb, **kwargs)
        center.scatter(m_theta, m_z, s=s, c=cm, **kwargs)

    pmt_type_mask = pmt_types == pmt_type
    chits = [cher_hits_s, cher_hits_l]
    shits = [scint_hits_s, scint_hits_l]
    hitres = [short_hit_time_res, long_hit_time_res]
    charges = [short_charges, long_charges]
    ctimes = [cher_times_s, cher_times_l]
    stimes = [scint_times_s, scint_times_l]
    ccharges = [cher_charges_s, cher_charges_l]
    scharges = [scint_charges_s, scint_charges_l]

    pmt_txt = ['Short', 'Long']
    theta, r = to_polar(ev.vertices[0].pos)
    theta = -theta[0]
    r = r[0]
    z = ev.vertices[0].pos[2]
    plt.text(0.12, 0.83, text,
             fontsize=20, transform=fig.transFigure)
    # plt.text(0.12, 0.79, f'$(r, \\theta, z)=({r/1000:.0f} m, {theta:.2f}, {z/1000:.0f} m)$', fontsize=8, transform=fig.transFigure)

    ztop = pmt_positions[pmt_type_mask][:, 2].max()
    zbottom = pmt_positions[pmt_type_mask][:, 2].min()
    plot_hits(pmt_positions[pmt_type_mask], ztop, zbottom,
              'grey', alpha=0.2, zorder=-999, marker='s')

    time_mask = hitres[pmt_type-1] < (3 if pmt_type == 2 and tcut else 30000)
    tresid_cut_fmt = ["All", "$t_{\\rm resid} < 3$ ns"]
    blue_cmap = get_continuous_cmap(
        "#6666ff #7575ff #8484ff #9393ff #a3a3ff #b2b2ff".split(' '))
    yellow_cmap = get_continuous_cmap(
        "#ffff66 #ffff75 #ffff84 #ffff93 #ffffa3 #ffffb2 #ffffc1 #ffffd1".split(" "))
    cmaps = [blue_cmap, yellow_cmap]

    if log:
        normdict = dict(norm=LogNorm())
    else:
        normdict = dict()
    plot_hits(hit_positions[hit_types == pmt_type][time_mask], ztop, zbottom, c=charges[pmt_type-1][time_mask], cmap='cool',
              alpha=1, s=2, zorder=0, marker='o', label=f"{pmt_txt[pmt_type-1]} Wavelengths ({tresid_cut_fmt[pmt_type-1]})",
              **normdict)
    plt.legend(frameon=False, bbox_to_anchor=(0.85, 3.23), loc='upper left', ncol=1, fontsize=12)
    plt.colorbar(center.get_children()[
                 1], ax=center, label="Photo-electrons (p.e.)", fraction=0.046, pad=0.0)

    # plot_hits(chits[pmt_type-1], ztop, zbottom,  alpha=1, s=5, marker='o', zorder=-997, label='Cherenkov Radiation')
    # plot_hits(shits[pmt_type-1], ztop, zbottom,  alpha=0.6, marker='o', zorder=-998, label='Scintillation')
    

    # ---------------------------------- 3d plot --------------------------------- #
    ax3d = fig.add_subplot(gridspec[2], projection='3d',)
    # ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # make the grid lines transparent
    # ax3d.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax3d.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax3d.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax3d.view_init()

    ax3d.scatter(*geo.channel_index_to_position.T,
                 alpha=0.1, color='grey', s=0.1)
    r, theta = to_polar(ev.vertices[0].pos)
    r = r[0]
    theta = -theta[0]
    ax3d.scatter(*ev.vertices[0].pos.T, s=5, color='cyan')
    zmin, zmax = geo.channel_index_to_position[:, 2].min(
    ), geo.channel_index_to_position[:, 2].max()
    ax3d.plot([0, ev.vertices[0].pos[0]], [0, ev.vertices[0].pos[1]], [
              zmin, zmin], color='grey', ls='--', lw=0.5, alpha=0.5)
    ax3d.plot([ev.vertices[0].pos[0]]*2, [ev.vertices[0].pos[1]]*2,
              [zmin, ev.vertices[0].pos[2]], color='grey', ls='--', lw=0.5)
    ax3d.scatter([0], [0], [zmin], color='black', s=10)

    X = 5
    r = X*400
    h = X*1000
    u = np.linspace(0, h, 20)
    theta = np.linspace(0, 2*np.pi, 20)
    uu, tt = np.meshgrid(u, theta)
    x = (h-uu)/h*r*np.cos(tt)
    y = (h-uu)/h*r*np.sin(tt)
    z = uu-h
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    rot = gen_rot([0, 0, -1], -ev.vertices[0].dir)
    cone_pts = (np.dot(rot, [x, y, z]).T+ev.vertices[0].pos).T
    ax3d.plot(*cone_pts, color='tab:blue', alpha=0.5)
    # ax3d.plot(*cone_pts[:2], [zmin]*len(cone_pts[0]), color='grey', linewidth=0.5)
    ax3d.set_xlabel('x'), ax3d.set_ylabel('y'), ax3d.set_zlabel('z')
    ax3d.set_xlim(-15_000, 15_000)
    ax3d.set_ylim(-15_000, 15_000)
    ax3d.set_zlim(-15_000, 15_000)
    ax3d.set_axis_off()

    # -------------------------------- plot props -------------------------------- #
    circ1.set_rmax(circ1.get_rmax()*0.96)
    circ2.set_rmax(circ2.get_rmax()*0.96)
    circ1.set_rticks([]), circ1.set_xticks([])
    circ2.set_rticks([]), circ2.set_xticks([])
    center.set_xticks([]), center.set_yticks([])
    center.set_xlim(*(0.92*np.array(center.get_xlim())))
    center.set_ylim(*(0.94*np.array(center.get_ylim())))
    circ1.grid(False)
    circ2.grid(False)
    # plt.colorbar()
    fig.subplots_adjust(hspace=-0.127)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def hist(data: List[h5py.Dataset] or h5py.Dataset, bins=None, xlabel=None, ylabel=None, title=None, log=False, error=True, grid=False, legend=True, color=None, norm=None, capsize=0, xlog=False, density=False, **kwargs):
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
            drawstyle='steps-mid', linewidth=plt.rcParams["axes.linewidth"], label=label, color=c)

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


def hist2d(data, bins=[100, 100], /, xlabel=None, ylabel=None, title=None, log=False, cmap="inferno", **kwargs):
    if log:
        Scale = LogNorm()
    else:
        Scale = None

    datax, datay = data
    h, x, y, _ = plt.hist2d(datax, datay, bins=bins)
    plt.pcolor(x, y, h.T, norm=Scale, cmap=cmap)
    plt.colorbar()

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=12)

    plt.show()


def hist3d(data, xbins, ybins=None, xlabel=None, ylabel=None, zlabel=None, title=None, log=False, cmap="inferno", contour=True, **kwargs):
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
