import os
from typing import List

from .data import DichroiconDataWriter
from .data import DummyDichroiconData
from .observables import THEIA_OBSERVABLES
from ..misc_utils import get_mask, dist_to_wall
from ..log import logger

import numpy as np
from numpy.typing import ArrayLike
from chroma.detector import Detector
from chroma.event import (
    NO_HIT,
    BULK_ABSORB,
    SURFACE_DETECT,
    SURFACE_ABSORB,
    RAYLEIGH_SCATTER,
    REFLECT_DIFFUSE,
    REFLECT_SPECULAR,
    SURFACE_REEMIT,
    SURFACE_TRANSMIT,
    BULK_REEMIT,
    CHERENKOV,
    SCINTILLATION,
    NAN_ABORT,
)

class Unpack:
    """Unpacker"""
    def __init__(self, detector: Detector or List[ArrayLike],
                 group_velocity: List[float] = [300/1.5]*2,
                 tts: float = 1.0,
                 filename: str or None = None,
                 overwrite_if_exists: bool = False,
                 verbose: bool = True):
        """Unpack constructor

        Parameters
        ----------
        detector : Detector or List[ArrayLike]
            The unpacker requires the positions and types of each channel in the Detector.
            This variable must be either the Detector object of a list containing
            the channel_index_to_position and channel_index_to_channel_type arrays in the
            form [channel_index_to_position, channel_index_to_channel_type].
        group_velocity (mm/ns): List[float], optional
            A list of group velocities for long and short wavelength photons to be used in
            calculating hit time residuals, by default a refractive index of 1.5 is used
            for both wavelength ranges.
        tts (ns): float, optional
            Transit-time-spread of photomultiplier tubes, by default 1.0 ns
        filename : str or None, optional
            Filename to save unpack data to. If not specified, a HDF5-like object
            is created in memory with groups "long" and "short" for and datasets
            cher_counts_d", "scint_counts_d", "tot_counts_d", "tresid_ch", "tresid_sc",
            and "tresid_tot".
            object
        overwrite_if_exists : bool, optional
            If an HDF5 file of the given filename already exists, completely overwrite the
            file. Otherwise, append to it. By default False
        verbose : bool, optional
            Verbose flag to input into `DichroiconDataWrtier`, by default True

        Raises
        ------
        TypeError
            _description_
        """
        if isinstance(detector, Detector):
            self.channel_index_to_channel_type = detector.channel_index_to_channel_type
            self.channel_index_to_position = detector.channel_index_to_position
        elif isinstance(detector, list) and len(detector) == 2:
            self.channel_index_to_position = detector[0]
            self.channel_index_to_channel_type = detector[1]
        else:
            raise TypeError("detector must be a Detector object or a list of channel types and positions.")
        
        self.ev_idx = 0
        self.gv_l, self.gv_s = group_velocity  # did you forget mm/ns !?!?!?!?
        self.tts = tts

        self.filename = filename
        if filename:
            filename = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace("recon", "cache"))
            self.dataset = DichroiconDataWriter(filename, THEIA_OBSERVABLES, overwrite_if_exists, verbose)
        else:
            self.dataset = DummyDichroiconData(select=["cher_counts_d", "scint_counts_d", "tot_counts_d",
                                                       "tresid_ch", "tresid_sc", "tresid_tot"])
        
    def digest(
        self,
        positions,
        times,
        true_pos,
        true_dir,
        true_time,
        flat_hits,
        charges,
        vertices=None,
        photons_beg=None,
        photon_tracks=None,
        dirfit_mask=None,
        type="",
        ):
        '''digest a single event
        
        Parameters
        ----------
        positions
            PMT positions
        times
            array of hit times
        true_pos
            position of the initial electron
        true_dir
            direction of the electron
        true_time
            time of initial electron
        flat_hits
            a list of photon objects that were detected by PMTs
        charges
            array of charges for each PMT hit
        vertices
            list of vertices
        photons_beg
            initial photons
        photon_tracks
            list of photon tracks
        dirfit_mask
            mask that indicates which hit_channel is a long or short PMT.
            used to discriminate between hits detected on long and short PMTs.
        type
            "long" or "short"
        '''        

        self.ev_idx += 1

        """
        General information saving for detected hits.
            - calculates and saves time residuals with/without a prompt cut
            - separates distributions of wavelengths, hit time residuals, and cos[alpha]'s in terms
              of photon type (Cherenkov, Scintillation, Reemission, and Total)
        
        """

        # first, calculate overall tresids and costhetas
        P = positions - true_pos
        D = np.sqrt(np.sum(np.square(P), axis=1))
        T = times - true_time
        tresid = T - D / (self.gv_l if type == "long" else self.gv_s)

        # deal with direction fit mask
        if dirfit_mask is None:
            cosalpha = np.sum(P * true_dir, axis=1) / D
        else:
            cosalpha = np.sum(P[dirfit_mask] * true_dir, axis=1) / D[dirfit_mask]
            tresid = tresid[dirfit_mask]

        self._add('e_pos', true_pos, type)
        self._add('e_dir', true_dir, type)
        self._add('e_t0', true_time, type)
        self._add("pmt_pos", positions[dirfit_mask], type)
        self._add("hit_charge", charges[dirfit_mask], type)
        
        # calculate distance to wall of detector (nb. takes ~60 ms because it's a
        # brute-force approximation)
        self._add("distance_to_detector_wall", dist_to_wall(true_pos, true_dir), type)

        # now, we'll calculate wavelengths, tresids, and costhetas with flag masks
        # ONLY takes into account detected photons.
        if flat_hits:
            # ls for long or short PMTs (distinction made in pyrat file)
            ls = flat_hits[dirfit_mask]

            # create masks for each type of flagged photon (we care about)
            ch_mask = get_mask(ls.flags, CHERENKOV | SURFACE_DETECT)
            ch_no_re_mask = get_mask(ls.flags, CHERENKOV | SURFACE_DETECT, none_of=BULK_REEMIT)
            ch_re_mask = get_mask(ls.flags, CHERENKOV | SURFACE_DETECT | BULK_REEMIT)
            sc_mask = get_mask(ls.flags, SCINTILLATION | SURFACE_DETECT)
            sc_no_re_mask = get_mask(ls.flags, SCINTILLATION | SURFACE_DETECT, none_of=BULK_REEMIT)
            sc_re_mask = get_mask(ls.flags, SCINTILLATION | SURFACE_DETECT | BULK_REEMIT)
            re_mask = get_mask(ls.flags, SURFACE_DETECT | BULK_REEMIT)

            # cherenkov photons [det], no reemission
            self._add("wv_ch_d", ls.wavelengths[ch_mask], type)
            self._add("tresid_ch", tresid[ch_mask], type)
            self._add("cosalpha_ch", cosalpha[ch_mask], type)
            self._add("wv_ch_no_reemit_d", ls.wavelengths[ch_no_re_mask], type)
            self._add("tresid_no_reemit_ch", tresid[ch_no_re_mask], type)
            self._add("cosalpha_no_reemit_ch", cosalpha[ch_no_re_mask], type)

            # scintillated photons [det]
            sc_wv = ls.wavelengths[sc_mask]
            sc_cos = cosalpha[sc_mask]
            self._add("tresid_sc", tresid[sc_mask], type)
            self._add("cosalpha_sc", sc_cos, type)
            self._add("wv_sc_d", sc_wv, type)
            self._add("tresid_no_reemit_sc", tresid[sc_no_re_mask], type)
            self._add("cosalpha_no_reemit_sc", cosalpha[sc_no_re_mask], type)
            self._add("wv_sc_no_reemit_d", ls.wavelengths[sc_no_re_mask], type)

            # reemitted photons [det]
            self._add("wv_re_d", ls.wavelengths[re_mask], type)
            self._add("cosalpha_re", cosalpha[re_mask], type)
            self._add("tresid_re", tresid[re_mask], type)

            # totals [det]
            # det isn't implied as there is also wv_tot_g
            self._add("wv_tot_d", ls.wavelengths, type)
            self._add("tresid_tot", tresid, type)  # det is implied
            self._add("cosalpha_tot", cosalpha, type)  # det is implied

            # counts [det]
            self._add("cher_no_reemit_d", np.count_nonzero(ch_no_re_mask), type)
            self._add("cher_reemit_d", np.count_nonzero(ch_re_mask), type)
            self._add("cher_counts_d", np.count_nonzero(ch_mask), type)
            self._add("scint_reemit_d", np.count_nonzero(sc_re_mask), type)
            self._add("scint_no_reemit_d", np.count_nonzero(sc_no_re_mask), type)
            self._add("scint_counts_d", np.count_nonzero(sc_mask), type)
            self._add("reemit_counts_d", np.count_nonzero(re_mask), type)
            self._add("tot_counts_d", ls.flags.shape[0], type)

            # try to detect leaks e.g. detected scintillation on longpass before 475 nm
            # if "long" in type and [wv for wv in sc_wv if wv < 475]:
            #     pos = [list(pos) for pos, wv, c in zip(positions, sc_wv, sc_cos) if wv < 475]
            #     self._add("wv475nm_sc_pos", np.asarray(true_pos), type)
            #     self._add("wv475nm_sc_dir", np.asarray(true_dir), type)
            #     if photon_tracks:
            #         leak_tracks = [photon for photon, wv in zip(photon_tracks, sc_wv) if wv < 450]

        if vertices is not bool and vertices is not None:
            """
            Calculate track length for initial Electron vertex.
            """
            # steps calc
            steps_x = vertices[0].steps.x
            steps_y = vertices[0].steps.y
            steps_z = vertices[0].steps.z
            dx = steps_x[1:] - steps_x[:-1]
            dy = steps_y[1:] - steps_y[:-1]
            dz = steps_z[1:] - steps_z[:-1]
            self._add(
                "steps",
                np.sum([np.sqrt(np.square(x) + np.square(y) * np.square(z)) for x, y, z in zip(dx, dy, dz)]),
                type,
            )
            """
            Grab position, energy, and number of electrons whose parents are gamma particles.
            """
            # traversal
            vertice = vertices[0]
            pos, ke = self.check_children(vertice.children, "gamma")
            if pos:
                self._add("children_pos", pos, type)
                self._add("children_ke", ke, type)
                self._add("children_num", len(pos), type)

        """
        Grab all initial photon distances from the initial electron position,
        as well as initial wavelengths for cherenkov, scintillation, reemission,
        and all photons!
            - discriminates between Cherenkov and Scintillation
            - includes ALL initial photons, not just ones that were detected by PMTs
        """
        if not isinstance(photons_beg, bool) and photons_beg is not None:
            beg_flags = photons_beg.flags
            beg_positions = photons_beg.pos
            ch_mask = get_mask(beg_flags, CHERENKOV, none_of=BULK_REEMIT)
            sc_mask = get_mask(beg_flags, SCINTILLATION, none_of=BULK_REEMIT)
            re_mask = get_mask(beg_flags, BULK_REEMIT)
            ch_pos = beg_positions[ch_mask]
            sc_pos = beg_positions[sc_mask]
            re_pos = beg_positions[re_mask]
            # ch_dist = [np.sqrt(np.square(x) + np.square(y) + np.square(z)) for [x, y, z] in ch_pos]
            # sc_dist = [np.sqrt(np.square(x) + np.square(y) + np.square(z)) for [x, y, z] in sc_pos]

            self._add("beg_ch_pos", ch_pos, type)
            self._add("beg_sc_pos", sc_pos, type)
            self._add("beg_re_pos", re_pos, type)

            if "long" in type:  # we don't want to save the same info twice
                # scintillation photons [gen]
                self._add("wv_ch_g", photons_beg.wavelengths[ch_mask], type)

                # scintillation photons [gen]
                self._add("wv_sc_g", photons_beg.wavelengths[sc_mask], type)

                # reemitted photons [gen] -- will be nothing if not scintllator medium like labppo
                self._add("wv_re_g", photons_beg.wavelengths[re_mask], type)

                # total emitted photons
                self._add("wv_tot_g", photons_beg.wavelengths, type)

        self._flush()

    def digest_event(self, ev):
        """digest_event
        
        Digest a single chroma Event.

        *** chroma_keep_flat_hits MUST be set to true in pyrat in order to use this
        unpacker. 
        
        optional pyrat simulation parameters include:
            - chroma_keep_photons_beg: save initial photon data
            - chroma_photon_tracking: save photon track data
            - chroma_particle_tracking: save particle tracking data

        Parameters
        ----------
        ev : chroma.Event
            Chroma event to digest.
        """
        
        if ev.flat_hits is None:
            logger.warning("No hits found in event. Is chroma_keep_flat_hits set to true?")
            return

        true_pos = ev.vertices[0].pos
        true_time = ev.vertices[0].t0
        true_dir = ev.vertices[0].dir
        # \/ this gets channel id's from each photon, so there'll be a ton of dupes
        hit_channels = ev.flat_hits.channel
        true_times = ev.flat_hits.t
        charges = ev.channels.q[hit_channels]
        
        times = np.random.normal(true_times, self.tts)  # apply transit time spread uncertainty to photons
        # detected positions set to PMT positions
        positions = self.channel_index_to_position[hit_channels]

        long_pmts = self.channel_index_to_channel_type==2
        dirfit_mask_l = long_pmts[hit_channels]
        dirfit_mask_s = (~long_pmts)[hit_channels]
        
        vertices = ev.vertices if ev.vertices[0].steps is not None else None
        
        # pass flat_hits to unpacker
        self.digest(
            positions,
            times,
            true_pos,
            true_dir,
            true_time,
            ev.flat_hits,
            charges,
            vertices,
            ev.photons_beg,
            ev.photon_tracks,
            dirfit_mask=dirfit_mask_l,
            type="long",
        )
        self.digest(
            positions,
            times,
            true_pos,
            true_dir,
            true_time,
            ev.flat_hits,
            charges,
            dirfit_mask=dirfit_mask_s,
            type="short",
        )
                
    def _add(self, group, data, type):
        assert isinstance(group, str), "[Unpack] First variable should be a string."

        self.dataset.add_to(group, data, type)

    # function to traverse child tree of a Vertex recursively
    def _check_children(
            self, children, particle_type="gamma", children_pos=[],
            children_ke=[],
            parent="e-"):
        if children:
            for child in children:
                if particle_type in parent:
                    # self._add("children_pos", child.pos)
                    # self._add("children_ke", child.ke)
                    children_pos.append(child.pos)
                    children_ke.append(child.ke)
                pos, ke = self._check_children(
                    child.children, child.particle_name, children_pos, children_ke)
            return pos, ke
        else:
            return children_pos, children_ke

    def _flush(self):
        self.dataset.flush()

    def close(self):
        self.dataset.close()
        # we delete the h5py object because it can't be pickled.
        del self.dataset

    def __repr__(self):
        closed = not hasattr(self, "dataset")
        if closed:
            return f'<Closed Unpack file>'
        
        class_str = self.dataset.__class__.__name__
        return f'<Unpack file "{self.filename}" ({class_str}) (mode {self.dataset.mode})>'
