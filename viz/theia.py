from typing import List

import matplotlib.pyplot as plt
import numpy as np
from geometry.theia import gen_rot
from matplotlib.colors import LogNorm
from numpy.typing import ArrayLike

from chroma.detector import Detector
from chroma.event import BULK_REEMIT, CHERENKOV, SCINTILLATION, SURFACE_DETECT, Event

from .utils import (dark_background, get_continuous_cmap, get_mask,
                    set_custom_style, to_polar, _particle_fmt, _si_prefixes)
from ..misc_utils import refractive_index_long, refractive_index_short

__all__ = ["event_display"]


@set_custom_style
@dark_background
def event_display(detector: Detector or List[ArrayLike],
                  ev: Event,
                  filename: str = None,
                  pmt_type: int = 1,
                  cmap="cool",
                  prompt_cut=False,
                  log=False):
    '''This function plots the event display of a given event.
    
    Parameters
    ----------
    detector : Detector or List[ArrayLike]
        The event display requires the positions and types of each channel in the Detector.
        This variable must be either the Detector object of a list containing
        the channel_index_to_position and channel_index_to_channel_type arrays in the
        form [channel_index_to_position, channel_index_to_channel_type].
    ev : chroma.Event
        Event to visualize
    filename : str, optional
        Filename to save the event display .png to,
    pmt_type : int, optional
        PMT type in the detector geometry. For Theia, 1 = short and 2 = long,
        default is 1.
    cmap : str, optional
        color map to use on PMT hits, by default "cool" (blue & pink)
    prompt_cut : float, optional
        Prompt cut to apply to long PMT hits. If False, no cut is applied. 
        False by defualt.
    log : bool, optional
        Whether to apply logarithmic scaling to the colormap. 
    '''
    
    
    if isinstance(detector, Detector):
        pmt_positions = detector.channel_index_to_position
        pmt_types = detector.channel_index_to_channel_type
    elif isinstance(detector, list) and len(detector) == 2:
        pmt_positions = detector[0]
        pmt_types = detector[1]
        
    
    zbottom_l, zbottom_s = pmt_positions[pmt_types == 2][:, 2].min(
    ), pmt_positions[pmt_types == 1][:, 2].min()
    ztop_l, ztop_s = pmt_positions[pmt_types == 2][:, 2].max(
    ), pmt_positions[pmt_types == 1][:, 2].max()

    """ ---------------------- gather info about the event --------------------- """
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

    # calculate hit time residuals
    long_hit_time_res = long_times - \
        np.sqrt(np.sum(
            np.square(long_hits-ev.vertices[0].pos), axis=1))/(300/refractive_index_long)
    short_hit_time_res = short_times - \
        np.sqrt(np.sum(
            np.square(short_hits-ev.vertices[0].pos), axis=1))/(300/refractive_index_short)

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
    
    
    # format title in corner
    particle_name = _particle_fmt[ev.vertices[0].particle_name]
    particle_MeV = ev.vertices[0].ke
    particle_eV = 1000000*particle_MeV    
    keys = list(_si_prefixes.keys())
    vals = list(_si_prefixes.values())
    (k,v), *__ = sorted(zip(keys,vals), key=lambda x: abs(x[1]-particle_eV))
    transformed_particle_energy = particle_eV/v
    if transformed_particle_energy == int(transformed_particle_energy) :  # e.g. 5.0 MeV
        particle_ke = f'{particle_eV/v:.0f} {k}eV'
    else:
        num_dec = str(particle_eV/v)[::-1].find('.')
        num_dec = min(num_dec, 2)
        particle_ke = f'{particle_eV/v:.{num_dec}f} {k}eV'

    text = f"{particle_name}\n{particle_ke}"
    plt.text(0.12, 0.83, text,
             fontsize=20, transform=fig.transFigure)
    # plt.text(0.12, 0.79, f'$(r, \\theta, z)=({r/1000:.0f} m, {theta:.2f}, {z/1000:.0f} m)$', fontsize=8, transform=fig.transFigure)

    ztop = pmt_positions[pmt_type_mask][:, 2].max()
    zbottom = pmt_positions[pmt_type_mask][:, 2].min()
    plot_hits(pmt_positions[pmt_type_mask], ztop, zbottom,
              'grey', alpha=0.2, zorder=-999, marker='s')

    time_mask = hitres[pmt_type-1] < (prompt_cut if pmt_type == 2 and prompt_cut is not None else np.inf)
    tresid_cut_fmt = ["All", "$t_{\\rm resid} < %0.1f$ ns"%prompt_cut]
    # blue_cmap = get_continuous_cmap(
    #     "#6666ff #7575ff #8484ff #9393ff #a3a3ff #b2b2ff".split(' '))
    # yellow_cmap = get_continuous_cmap(
    #     "#ffff66 #ffff75 #ffff84 #ffff93 #ffffa3 #ffffb2 #ffffc1 #ffffd1".split(" "))
    # cmaps = [blue_cmap, yellow_cmap]

    if log:
        normdict = dict(norm=LogNorm())
    else:
        normdict = dict()
    plot_hits(hit_positions[hit_types == pmt_type][time_mask], ztop, zbottom, c=charges[pmt_type-1][time_mask], cmap=cmap,
              alpha=1, s=2, zorder=0, marker='o', label=f"{pmt_txt[pmt_type-1]} Wavelengths ({tresid_cut_fmt[pmt_type-1]})",
              **normdict)
    plt.legend(frameon=False, bbox_to_anchor=(0.85, 3.23), loc='upper left', ncol=1, fontsize=12)
    plt.colorbar(center.get_children()[
                 1], ax=center, label="Photo-electrons (p.e.)", fraction=0.046, pad=0.0)

    # plot_hits(chits[pmt_type-1], ztop, zbottom,  alpha=1, s=5, marker='o', zorder=-997, label='Cherenkov Radiation')
    # plot_hits(shits[pmt_type-1], ztop, zbottom,  alpha=0.6, marker='o', zorder=-998, label='Scintillation')
    

    # ---------------------------------- 3d plot --------------------------------- #
    ax3d = fig.add_subplot(gridspec[2], projection='3d',)
    # plot detectors
    ax3d.scatter(*pmt_positions[pmt_type_mask].T,
                 alpha=0.1, color='grey', s=0.5)
    
    # plot event vertex & line indicators around it
    r, theta = to_polar(ev.vertices[0].pos)
    r = r[0]
    theta = -theta[0]
    ax3d.scatter(*ev.vertices[0].pos.T, s=5, color='white', zorder=999)
    zmin, zmax = pmt_positions[:, 2].min(
    ), pmt_positions[:, 2].max()
    ax3d.plot([0, ev.vertices[0].pos[0]], [0, ev.vertices[0].pos[1]], [
              zmin, zmin], color='grey', ls='--', lw=0.5, alpha=1, zorder=999)
    ax3d.plot([ev.vertices[0].pos[0]]*2, [ev.vertices[0].pos[1]]*2,
              [zmin, ev.vertices[0].pos[2]], color='grey', ls='--', lw=0.5, zorder=999)
    ax3d.scatter([0], [0], [zmin], color='black', s=10, zorder=999)

    # plot Cerenkov cone in direction of vertex
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
    ax3d.plot(*cone_pts, color='lightgrey', alpha=0.5, zorder=999)
    
    # plot hits if we're looking at long PMT hits
    if pmt_type == 2:
        _cmap = center.get_children()[1].get_cmap()
        ax3d.scatter(*hit_positions[hit_types == pmt_type][time_mask].T,
                    c=charges[pmt_type-1][time_mask],
                    cmap=_cmap,
                    alpha=0.5, s=0.5)
    
    # axis housekeeping
    ax3d.set_xlabel('x'), ax3d.set_ylabel('y'), ax3d.set_zlabel('z')
    ax3d.set_xlim(-15_000, 15_000)  # TODO: make this dynamic, depending on radius of detector
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
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

