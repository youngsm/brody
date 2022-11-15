from typing import List
import re
import tempfile
import os

import h5py
import numpy as np

from .observables import Observable, Observable3D, ObservableLike

from ..log import logger
from . import utils

""" DichroiconData is a data-saving class that is called after each event in a pyrat simulation.
It is a (thin) wrapper around the normal h5py.File class.

A cache hdf5 file is created at initialization, and data is incrementally written as the simulation
continues.

It should be initialized with an filename string (what to save the h5 file as) and a list of Observable's.
Observables are the things that are being measured, and are classes that contain a few key variables that describe
what sort of data DichroiconData will be working with.

To add data, call DichroiconData's add_to() function, inputting parameters name, which is the name of the dataset
to which you're adding to, a numpy array of data to add to the existing dataset, and which PMT type the data came from
("long" or "short").

Your data must be serializable to be saved.

After each event, flush the data to the cache file by calling DichroiconData's flush() function.

At the end of the simulation, close the h5py file by calling Dset.close(). This is super simple and closes the current open file.
"""

class File(h5py.File):
    """Wrapper around h5py.File that adds the ability to 'add' two files together."""

    def __add__(self, other):
        return self.combine([self.filename, other.filename], temp=True)

    @classmethod
    def combine(cls, filenames: List[str],
                out: str = None,
                link: bool = True,
                temp: bool = False,
                return_file: bool = False):
        """Combine N datasets of the same architecture into a single dataset

        Note that this function does NOT check for the same architecture,
        so be careful.

        Important: a Virtual Dataset is created, so the data is not actually
        found in the resulting file -- it's just a pointer. So DO NOT
        delete the original files!

        Parameters
        ----------
        fnames : List[str]
            list of filenames to combine
        link : bool, optional
            whether to create a virtual dataset (i.e., a pointer to the datasets found in
            fnames) or an actual copy of the data, by default True
        out : str, optional
            output filename, by default None

        Returns
        -------
        str
            filename of the combined dataset
        """
        def dataset_info(f: h5py.Dataset, key: str) -> dict:
            if isinstance(f[key]._id, h5py.h5d.DatasetID):
                curr = f[key]
                return {key: [curr.shape, curr.dtype, curr.attrs]}
            elif isinstance(f[key]._id, h5py.h5g.GroupID):
                if len(f[key].keys()) == 0:
                    return {}
                d1, *other = [dataset_info(f, f"{key}/{_key}") for _key in f[key].keys()]
                for d2 in other:
                    d1.update(d2)
                return d1
            else:
                raise ValueError('Unknown id type: {}'.format(type(f[key]._id)))

        # -------------------------------- type checks ------------------------------- #
        if not np.iterable(filenames):
            raise TypeError("filenames must be iterable!")

        if all(map(lambda x: isinstance(x, str), filenames)):
            opened_files = [h5py.File(f, "r") for f in filenames]
        elif all(map(lambda x: isinstance(x, h5py.File), filenames)):
            opened_files = list(filenames)
        else:
            raise TypeError(
                "filenames must be a list of strings or a list of h5py.File objects")

        # -------------------------- initialize output file -------------------------- #
        if out is None:
            out = re.sub(r"_GPU\d+", "", filenames[0])
            logger.info('Saving combined datasets to %s', out)
        if temp:
            tf = tempfile.NamedTemporaryFile()
            out = tf
            return_file = True
        outf = h5py.File(out, "w")

        if link:
            # first, gather some info about the datasets
            described_datasets = []
            for f in opened_files:
                desc = {}
                for k in f.keys():
                    desc.update(dataset_info(f, k))
                described_datasets.append(desc)

            # aggregate this data into a single dictionary
            totals = {}
            for f in described_datasets:
                for k in f:
                    if k not in totals:
                        totals[k] = f[k]
                        totals[k][0] = [totals[k][0]]
                    else:
                        totals[k][0].append(f[k][0])

            for k in totals:
                # create a virtual layout for each dataset/group pair
                shapes, dtype, attrs = totals[k]
                shape_c = (np.array(shapes)[:, 0].sum(), *shapes[0][1:])
                layout = h5py.VirtualLayout(shape=shape_c, dtype=dtype)

                # add in chunks between datasets
                i0 = 0
                for i in range(len(shapes)):
                    vsource = h5py.VirtualSource(opened_files[i].filename, k, shape=shapes[i])
                    layout[i0:shapes[i][0] + i0, ...] = vsource

                    i0 += shapes[i][0]
                outf.create_virtual_dataset(k, layout, fillvalue=0)  # is 0 an unsigned int?

                if attrs:
                    outf[k].attrs.update(attrs)

        else:  # don't use this -- it's slow as hell and rids you of hdf5's linker magic
            ff = opened_files[0]

            def copy_dataset(key):
                if isinstance(ff[key]._id, h5py.h5d.DatasetID):
                    outf.create_dataset(key,
                                        data=np.concatenate([f[key][...] for f in opened_files]),
                                        dtype=ff[key].dtype)
                    outf[key].attrs.update(ff[key].attrs)
                elif isinstance(ff[key]._id, h5py.h5g.GroupID):
                    for _key in ff[key]:
                        copy_dataset(f'{key}/{_key}')
                else:
                    raise ValueError('Unknown id type: {}'.format(type(ff[key]._id)))

            for k in ff.keys():
                copy_dataset(k)

        for f in opened_files:
            f.close()

        if return_file:
            return outf
        else:
            outf.close()
            return out
        
    @staticmethod
    def describe_file(file, key=None, max_depth=np.inf):
        obj = file[key] if key is not None else file
        print(utils.group_to_str(obj, max_depth=max_depth), end='')
        
    def describe(self, key=None, max_depth=np.inf):
        obj = self[key] if key is not None else self
        print(utils.group_to_str(obj, max_depth=max_depth), end='')
        
    def __repr__(self):
        if not self.__bool__():  # if closed
            return f"<Closed {self.__class__.__name__} file>"
        return utils.group_to_str(self, max_depth=1)
    
    if utils.H5Glance is not None:
        def _repr_html_(self):
            if not self.__bool__():
                from htmlgen import Division
                return str(Division(f"<Closed {self.__class__.__name__} file>"))
            return utils.H5Glance(self)._repr_html_()
    else:
        utils._warn_import()
    
class DichroiconData(File):
    """DichroiconData

    A wrapper around the h5py.File class that adds some functionality
    helpful to Dichroicon studies

    """

    def add_to(self, name, data, type):
        assert self._is_writable(), "[D-D] h5 file must be open and ready to be written to."

        array = np.asarray(data)
        d = self[f"{type}/{name}"]

        if len(array.shape) > 0 and array.shape[0] > 0:
            d.resize(d.shape[0] + array.shape[0], axis=0)
            d[-len(array):] = array
        elif array.shape == ():
            d.resize(d.shape[0] + 1, axis=0)
            d[-1] = array
        elif self.verbose:
            # happens when there's actually no data in the data array. very small
            # likelihood, but it happens e.g., low coverage
            logger.warning("[D-D] WARNING: NO DATA IN ADD_TO CALL FOR %s/%s", type.upper(), name.upper())

    def _is_open(self):
        return self.__bool__()

    def _is_readable(self):
        return self._is_open() and self.mode == "r"

    def _is_writable(self):
        return self._is_open() and self.mode == "r+"


class DichroiconDataWriter(DichroiconData):
    """DichroiconDataWriter

    A wrapper around the h5py.File class that adds some functionality
    helpful to Dichroicon studies

    """

    def __init__(self, filename, data: List[Observable] or None = None, verbose=False):
        self.verbose = verbose
        if os.path.exists(filename):
            super().__init__(filename, "r+")
        elif data is not None:
            super().__init__(filename, "w")
            self._init_groups(data)            
        else:
            raise ValueError("File needs to be initiated, but no list of observables were provided!")
                    
    def _init_groups(self, data: List[Observable]):
        self.create_group("long")
        self.create_group("short")
        # self.create_group("all")
        compression_kwargs = dict(compression="gzip")

        # initialize observables as datasets in their respective groups (long and short)
        for group in ["long", "short"]:
            for d in data:
                try:
                    self[group].create_dataset(
                        d.name,
                        d.init_shape,
                        d.dtype,
                        maxshape=d.maxshape,
                        chunks=True,
                        **compression_kwargs)
                    self[f"{group}/{d.name}"].attrs["description"] = d.description
                except RuntimeError as e:
                    raise RuntimeError("[D-D] Error while creating dataset {}/{}: {}".format(group, d.name, e)) from e

class DichroiconDataReader(DichroiconData):
    def __init__(self, filename, verbose=False):
        super().__init__(filename, "r")
        self.names = list(self["long"].keys())
        if verbose:
            logger.log(1, "[D-D] Opened h5 file: {}".format(self))
        # names must be a list of strings

    def grab(self, type, mask=None, names=None):
        assert self.__bool__(
        ), "[DSET] h5 file must be open and ready to be read from."

        if names is None:
            names = self.names
        names = np.atleast_1d(names)

        if mask is None:
            mask = np.ones_like(names, dtype=bool)

        dataset_path = type + "/{}"

        if len(names) == 1:
            return self[dataset_path.format(names[0])][mask]
        else:
            return {n: self[dataset_path.format(n)][mask] for n in names}


class DummyDichroiconData(dict):
    def __init__(self, select=None):
        super().__init__()
        self.select = select

    def add_to(self, group, data, type):
        if self.select is not None and group not in self.select:
            return
        self[f"{type}/{group}"] = data

    def reset(self):
        for k in self:
            del self[k]

    def flush(self):
        pass

    def close(self):
        self.reset()

    def as_dict(self):
        return dict(self)
    
    def __repr__(self):
        return "DummyDichroiconData({}, select={})".format(self.as_dict(), self.select)
