import numpy as np
from dataclasses import dataclass

__exports__ = ['THEIA_OBSERVABLES', 'Observable', 'Observable3D', 'ObservableLike']

@dataclass
class Observable:
    name: str
    dtype: str = "float32"
    description: str = ""
    init_shape: tuple = (0,)
    maxshape: tuple = (None,)

class Observable3D(Observable):
    def __init__(self, name, dtype=None, description=""):
        init_shape = (0, 3)
        maxshape = (None, 3)
        super().__init__(name, dtype, description, init_shape, maxshape)


class ObservableLike(Observable):
    def __init__(self, name, other, dtype=None, description=""):

        if hasattr(other, "__len__"):
            __, *off_axis = np.shape([other])
            init_shape = (0, *off_axis)
            maxshape = (None, *off_axis)
        else:
            init_shape = (0,)
            maxshape = (None,)

        if dtype is None:
            # get the type of what sort of object we're dealing with
            # in the data
            dtype = np.min_scalar_type(other)

        super().__init__(name, dtype, description, init_shape, maxshape)

THEIA_OBSERVABLES = [
# wavelengths generated and detected
Observable("wv_ch_g",               np.float32, "generated cherenkov wavelengths"),
Observable("wv_ch_d",               np.float32, "detected cherenkov (no reemission) wavelengths"),
Observable("wv_ch_no_reemit_d",     np.float32, "detected cherenkov wavelengths"),
Observable("wv_sc_g",               np.float32, "total detected counts"),
Observable("wv_sc_d",               np.float32, "detected scintillation wavelengths"),
Observable("wv_sc_no_reemit_d",     np.float32, "detected scintillation (no reemission) wavelengths"),
Observable("wv_re_g",               np.float32, "generated reemission wavelengths"),
Observable("wv_re_d",               np.float32, "detected reemission wavelengths"),
Observable("wv_tot_g",              np.float32, "generated total wavelengths"),
Observable("wv_tot_d",              np.float32, "detected total wavelengths"),
# cosalphas
Observable("cosalpha_ch",           np.float32, "cherenkov cosalpha"),
Observable("cosalpha_sc",           np.float32, "scintillation cosalpha"),
Observable("cosalpha_no_reemit_ch", np.float32, "cherenkov (no reemission) cosalpha"),
Observable("cosalpha_no_reemit_sc", np.float32, "scintillation (no reemission) cosalpha"),
Observable("cosalpha_re",           np.float32, "reemission cosalpha"),
Observable("cosalpha_tot",          np.float32, "total cosalpha"),
# time residuals
Observable("tresid_ch",             np.float32, "cherenkov time residuals"),
Observable("tresid_sc",             np.float32, "scintillation time residuals"),
Observable("tresid_no_reemit_ch",   np.float32, "cherenkov (no reemision) time residuals"),
Observable("tresid_no_reemit_sc",   np.float32, "scintillation (no reemission) time residuals"),
Observable("tresid_re",             np.float32, "reemission time residuals"),
Observable("tresid_tot",            np.float32, "total time residuals"),
# ===== detection counts ======
# cherenkov
Observable("cher_no_reemit_d",      np.uint32, "detected cherenkov (no reemit)"),
Observable("cher_reemit_d",         np.uint32, "detected cherenkov (reemit only)"),
Observable("cher_counts_d",         np.uint32, "detected cherenkov (all)"),

# scintillation
Observable("scint_no_reemit_d",     np.uint32, "detected scintillation (no reemission)"),
Observable("scint_reemit_d",        np.uint32, "detected scintillation (reemission only)"),
Observable("scint_counts_d",        np.uint32, "detected scintillation (all)"),

# reemitted
Observable("reemit_counts_d",       np.uint32, "total detected reemitted photons"),
# total
Observable("tot_counts_d",          np.uint32, "total detected photons"),
# ==============================
# other misc stuff
Observable("distance_to_detector_wall", np.float32, "distance to detector along the vertex of the initial event particle"),
Observable("hit_charge",            np.float32, "charges of hit PMTs"),
Observable("steps",                 np.float32, "path length of initial electron"),
Observable("children_ke",           np.float32, "kinetic energies of initial electron's children"),
Observable("children_num",          np.uint32, "number of children"),
Observable3D("pmt_pos",             np.float32, "positions of hit PMTs"),
Observable3D("children_pos",        np.float32, "positions of initial electron's children"),
Observable3D("beg_ch_pos",          np.float32, "init positions of cherenkov (no reemit) photons"),
Observable3D("beg_sc_pos",          np.float32, "init positions of scintillation photons"),
Observable3D("beg_re_pos",          np.float32, "init positions of remission photons"),
Observable3D("wv475nm_sc_dir",      np.float32, "direction of scintillation photons at or before 475nm"),
Observable3D("wv475nm_sc_pos",      np.float32, "position of scintillation photons at or before 475nm"),
Observable3D("e_pos",               np.float32, "initial electron position"),
Observable3D("e_dir",               np.float32, "initial electron position"),
Observable("e_t0",                  np.float32, "initial electron time"),
]
