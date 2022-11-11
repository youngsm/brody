import os

from .data import DichroiconData
from .data import DichroiconDataReader
from .data import DummyDichroiconData
from .constants import *
from ..misc_utils import get_mask, dist_to_wall
import numpy as np


class Unpack:
    def __init__(self, group_velocity, prompt_cut, filename=None, verbose=True):

        filename = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace("recon", "cache")) if filename is not None else None
        self.prompt_cut_mask = []
        self.ev_idx = 0
        self.group_velocity = group_velocity  # did you forget mm/ns !?!?!?!?
        self.prompt_cut = prompt_cut
        self.filename = filename
        if filename:
            if not os.path.exists(filename):
                # create datasets
                self.dataset = DichroiconData(filename, THEIA_OBSERVABLES, verbose)
            else:
                self.dataset = DichroiconDataReader(filename, verbose=True)
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

        self.ev_idx += 1
        """
        Functions!
            - get_mask: creates a boolean mask with a given array of flags and flag to check
            - check_children: traverses the tree of children from a vertice and returns positions and kinetic energies of
                              every gamma particle whose parent is an electron
        """

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
        tresid = T - D / self.group_velocity[0 if type == "long" else 1]

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
        self._add("distance_to_detector_wall", dist_to_wall(true_pos, true_dir))

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

    def digest_event(self, ev, db):
        true_pos = ev.vertices[0].pos
        true_time = ev.vertices[0].t0
        true_dir = ev.vertices[0].dir
        if db.chroma_keep_flat_hits:
            # \/ this gets channel id's from each photon, so there'll be a ton of dupes
            hit_channels = ev.flat_hits.channel
            true_times = ev.flat_hits.t
            charges = ev.channels.q[hit_channels]
            times = np.random.normal(true_times, db.theia_pmt_tts)  # apply transit time spread uncertainty to photons
            positions = db.det.channel_index_to_position[hit_channels]  # detected positions set to PMT positions

            photons_beg = ev.photons_beg if db.chroma_keep_photons_beg else None
            vertices = ev.vertices if db.chroma_particle_tracking else None
            photons_end = ev.photons_end if db.chroma_keep_photons_end else None
            photon_tracks = ev.photon_tracks if db.chroma_photon_tracking else None
            if "dichroicon" in db.theia_pmt:
                # pass flat_hits to unpacker
                self.digest(
                    positions,
                    times,
                    true_pos,
                    true_dir,
                    true_time,
                    ev.flat_hits,
                    charges,
                    vertices=vertices,
                    photons_beg=photons_beg,
                    photon_tracks=photon_tracks,
                    dirfit_mask=db.dirfit_mask_l[hit_channels],
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
                    dirfit_mask=db.dirfit_mask_s[hit_channels],
                    type="short",
                )
            else:
                self.digest(
                    positions,
                    times,
                    true_pos,
                    true_dir,
                    true_time,
                    ev.flat_hits,
                    charges,
                    vertices=vertices,
                    photons_beg=photons_beg,
                    photon_tracks=photon_tracks,
                    dirfit_mask=db.dirfit_mask[hit_channels],
                    type="long",
                )

    def __call__(self, func):
        assert not hasattr(self, "dataset"), "Dataset must be closed after being written to in order to be read."
        assert not isinstance(self, dict), "Dataset file name must be specified."
        def _unpack(db):
            self.digest_event(db.ev, db)
            return func(db)
        return _unpack
    
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
