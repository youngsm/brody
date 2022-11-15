from .h5glance.h5glance.terminal import group_to_str
try:
    from .h5glance.h5glance.ipython import H5Glance
except ImportError:
    H5Glance = None

from ...log import logger
def _warn_import():
    logger.warning(" htmlgen not installed; for fancy representations of HDF5 files, install it with `pip install htmlgen`")