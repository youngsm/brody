import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from .utils import polyform

class LikelihoodException(Exception):
    pass


class NLL1D(object):
    """
    A callable object that evaluates the likelihood values given some binned data

    Does a few nonstandard things:
        - Forces a 'min_val' for each bin to avoid -log(0) to make low stats behave and avoid NaNs
        - Rolls off the probability at PDF boundaries to 'pull' outliers into the defined range
            (probability e^(-Ax) with x being distance from pdf boundary and A being 'pull')

    If you don't specify a min_val, this will guess one as being 100 times less than the bin
    with the most counts.

    If you don't specify a pull, this will guess one by calculating the average A using at most
    10 points on each side of the data range passing the min_val cut.
    """

    def __init__(self, counts, edges, min_val=None, pull=None):
        """
        Set pull to None to estimate exponential falloff from data (if desired)
        Set min_val high enough to avoid poisson fluctuations near edges of PDF (if desired)
        """

        pdf = counts.copy()
        # find bounds on PDF where counts are above the minimum value
        if min_val is None:
            min_val = np.max(pdf) / 100
        pdf = np.where(pdf > min_val, pdf, min_val)
        pull_bounds = np.argwhere(pdf > min_val)
        left, right = pull_bounds[0][0], pull_bounds[-1][0]
        # if (right - left) < 20:
        #     raise LikelihoodException("Too few bins remaining for a good PDF (bin finer?)")

        # apply cut from above to the PDF
        pdf = pdf[left:right]
        widths = (edges[1:] - edges[:-1])[left:right]
        # calculate norm and normalize the PDF to 1
        norm = np.sum(pdf * widths)
        pdf = pdf / norm
        self.centers = ((edges[:-1] + edges[1:]) / 2)[left:right]

        # convert to NLL
        self.nll = -np.log(pdf)
        logged_min_val = -np.log(min_val / norm)
        self.nll[self.nll > logged_min_val] = logged_min_val

        if pull is None:
            npts = min(len(self.nll) // 2, 10)
            # fmt: off
            left_pull  = -np.mean((self.nll[:npts-1]-self.nll[1:npts])/(self.centers[:npts-1]-self.centers[1:npts]))
            right_pull =  np.mean((self.nll[-npts:-1]-self.nll[-npts+1:])/(self.centers[-npts:-1]-self.centers[-npts+1:]))
            self.pull  =  (left_pull, right_pull)
            # fmt: on
        else:
            self.pull = (pull, pull)
            
        self._call_fn = np.interp
        self._call_args = (self.centers, self.nll)

    def __call__(self, x, apply_pull=True):
        # this interp returns self.nll[0] and self.nll[-1] at values more or less than the centers
        # it is thus necessary to roll off probabilities at the edges to avoid degeneracies
        vec = self._call_fn(x, *self._call_args)
        if apply_pull:
            vec = self.apply_pull(x, vec)
        return vec

    def apply_pull(self, x, vec):
        """
        Roll off probabilities for x values past PDF boundaries
        """
        left_pull, right_pull = self.pull
        greater = x > self.centers[-1]
        deltax = x[greater] - self.centers[-1]
        vec[greater] = self.nll[-1] + deltax * right_pull
        lesser = x < self.centers[0]
        deltax = self.centers[0] - x[lesser]
        vec[lesser] = self.nll[0] + deltax * left_pull
        return vec

    def mean(self):
        weights = np.exp(-self.nll)
        return np.sum(self.centers * weights) / np.sum(weights)
    
    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(self.centers, self.nll, c='k', lw=0.5, label='data', zorder=-999)
        ax.plot(self.centers, self(self.centers), c='r', lw=0.5, label='fit')
        ax.set_xlabel("$x$")
        ax.set_ylabel("-log(likelihood)")
        ax.set_xlim(self.centers[0], self.centers[-1])
        ax.legend(frameon=False)
        ax.set_title(self.__class__.__name__)
        return ax




class CosAlphaNLL(NLL1D):
    """
    A callable object that evaluates the -log(likelihood) values by fitting
    Nth degree polynomials above and below the cherenkov peak of binned data
    """

    def __init__(self, counts, edges, order=5, *args, **kwargs):
        super().__init__(counts, edges, *args, **kwargs)

        cher_peak = self.centers[np.argmin(self.nll)]
        poly_right = np.zeros(order)
        poly_left = np.zeros(order)

        p, __ = curve_fit(polyform, self.centers, self.nll, p0=np.concatenate([[cher_peak], poly_left, poly_right]))

        self._call_fn = polyform
        self._call_args = p

class NLL2D(object):
    """
    A callable object that evaluates the -log(likelihood) values given two axes of some binned data
    Does a few nonstandard things:
        - Forces a 'min_val' for each bin to avoid -log(0) to make low stats behave and avoid NaNs
        - Rolls off the probability at PDF boundaries to 'pull' outliers into the defined range
            (probability e^(-Ax) with x being distance from pdf boundary and A being 'pull').

    Some nonstandard variables:
    - pull_axes: determines which axes whereby probabilities are rolled off via single string
        containing x and/or y or neither. Only supports axes, and not individual sides of the PDF.
    - pullsize: small, medium, or large. Determines maximum percentage of each axis that can be
                rolled off. 10% for small, 16% for medium, 20% for large.
    - verbose: saves a plot of the evaluated NLL.
    - pull: if specificed, is [xpull,ypull]

    If you don't specify a min_val, this will guess one as being 100 times less than the bin
    with the most counts.

    If you don't specify a pull, this will guess one by calculating the average A using at most
    10 points on each side of the data range passing the min_val cut.

    Note that the "counts" and "nll" arrays are treated as such:

    nll[0, -1] ------------------- nll[-1, -1]
              |                  |     ^           ycenters[-1]
              |                  |     |
              |    NLL  ARRAY    | ycenters
              |                  |     |
              |                  |     v           ycenters[0]
    nll[0, 0] ------------------- nll[0, -1]
               <--- xcenters --->

     xcenters[0]                xcenters[-1]

    where xcenters = (xedges[1:]+xedges[:-1])/2
    and likewise for ycenters

    """

    def __init__(self):
        return
    #     self,
    #     counts,
    #     xedges,
    #     yedges,
    #     min_val=None,
    #     pull_axes="",
    #     pull=None,
    #     verbose=False,
    #     s=0,
    # ):
    #     """
    #     1. Get data ready for manipulation
    #     """
    #     # hard edges
    #     xs = (xedges[:-1] + xedges[1:]) / 2
    #     ys = (yedges[:-1] + yedges[1:]) / 2
    #     self.xmin, self.xmax = xs[0], xs[-1]
    #     self.ymin, self.ymax = ys[0], ys[-1]

    #     # allow for capitals in pull_axes
    #     pull_axes = pull_axes.lower() if isinstance(pull_axes, str) else ""

    #     # length - 1 because we're dealing with edges and not centers
    #     # if for some reason the 2D counts array was transposed before calling NLLPDF_2D, this takes care of that.
    #     if len(counts) != len(xedges) - 1:
    #         counts = counts.T

    #     # copy counts to a new array and find the lengths of x and y axes
    #     self.nll = counts.copy()
    #     # find min_val
    #     if min_val is None and np.min(self.nll) == 0:
    #         min_val = np.max(self.nll) / 100
    #     elif np.min(self.nll) != 0:
    #         min_val = np.min(self.nll)

    #     # xmin, xmax, ymin, ymax = self.determine_edges(min_val, pull_axes, pullsize)
    #     xmin, xmax, ymin, ymax = 0, -1, 0, -1

    #     """
    #     3. Convert to NLL and centers along each axis within the pull bounds, and plot it (if specified)
    #     """
    #     sample_count = np.sum(self.nll)
    #     bin_area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])  # ASSUMES UNIFORM BIN AREAS!
    #     norm = sample_count * bin_area
    #     bbox = [min(xs), max(xs), min(ys), max(ys)]

    #     self.nll = RectBivariateSpline(xs, ys, self.nll / norm, bbox, kx=1, ky=1, s=0.0)(xs, ys)
    #     self.nll[self.nll < min_val / norm] = min_val / norm
    #     self.nll = -np.log(self.nll[xmin:xmax, ymin:ymax])
    #     self.min_val = np.min(self.nll)
    #     self.xcenters = xs[xmin:xmax]
    #     self.ycenters = ys[ymin:ymax]
    #     """
    #     4. Calculate pull along each side of the PDF.
    #     """
    #     self.pull = self.determine_pull(xmin, xmax, ymin, ymax) if pull is None else pull

    #     """ 
    #     (5.) Optionally fit the cos[theta] distribution in each time residual bin, sample the distribution, and set that as the
    #     NLL.
    #     """
    #     if polyform_fit:

    #         order = 5

    #         for idx, x in enumerate(self.xcenters):
    #             if abs(x) < 10:
    #                 nll = self.nll[idx, :]

    #                 cher_peak = self.ycenters[np.argmin(nll)]
    #                 poly_right = np.zeros(order)
    #                 poly_left = np.zeros(order)

    #                 p, p_cov = curve_fit(
    #                     polyform,
    #                     self.ycenters,
    #                     nll,
    #                     p0=np.concatenate([[cher_peak], poly_left, poly_right]),
    #                     maxfev=100000,
    #                 )
    #                 self.nll[idx, :] = polyform(self.ycenters, *p)

    #     """
    #     scipy.interpolate.RectBivariateSpline is a faster version of scipy.interpolate.interp2d
    #     Must switch y's and x's for this class (mirrors some MATLAB stuff)... or not? I suppose
    #     the transposed counts array at the start fixes this.
    #     kx and ky are polynomial powers; 1 for linear (what we want)
    #     s denotes smoothing -- we don't like smoothing
    #     """

    #     rbf = RectBivariateSpline(self.xcenters, self.ycenters, self.nll)
    #     self.interp = rbf
    #     # self.interp = interp2d(self.xcenters, self.ycenters, self.nll.T)

    #     if verbose:
    #         self.debug_plot(self.nll)

    # def determine_edges(self, min_val, pull_axes, pullsize):

    #     # find pull bounds (will be modified if needed later)
    #     pull_bounds = np.argwhere(self.nll > min_val)
    #     """
    #     2. Find top left, top right, bottom left, and bottom right indices
    #        of the pull_bounds mask.
    #        This traces out the largest possible rectangle whereby the value
    #        of each indice is True.
        
    #     """
    #     xlength = len(self.nll)
    #     ylength = len(self.nll[0])

    #     bottom_left = pull_bounds[0]
    #     top_right = pull_bounds[-1]

    #     # create a mask that returns the leftmost column.
    #     max_bottom_left = pull_bounds[:, 0] == bottom_left[0]
    #     # find last element of 1st column -- yields args of top-left point
    #     top_left = pull_bounds[max_bottom_left][-1]

    #     # create a mask that returns the rightmost column.
    #     max_bottom_right = pull_bounds[:, 0] == top_right[0]
    #     # find last element of 1st column -- yields args of top-right point
    #     bottom_right = pull_bounds[max_bottom_right][0]

    #     # take care of pull_axes, while ensuring pull_axes' requirements
    #     xmin = max(bottom_left[0], top_left[0]) if "x" in pull_axes else 0
    #     xmax = min(bottom_right[0], top_right[0]) if "x" in pull_axes else xlength - 1
    #     ymin = max(bottom_left[1], bottom_right[1]) if "y" in pull_axes else 0
    #     ymax = min(top_left[1], top_right[1]) if "y" in pull_axes else ylength - 1
    #     """
    #     We don't want to roll off more than ~10% of of binning on a given axis (5% above, 5% below),
    #     possibly getting rid of important structure.
        
    #     If the cut data outside of the pull_bounds exceeds this percentage of the array, the cut data
    #     is decreased to at most 5%, 8% or 10% on each side of the PDF.
    #     """
    #     max_difs = {
    #         "small": 0.05,
    #         "mid": 0.08,
    #         "large": 0.10,
    #     }  # to convert from string to size
    #     max_dif = max_difs[pullsize]  # in percentage
    #     if (ylength - ymax) > max_dif * ylength:
    #         ymax = round((1 - max_dif) * (ylength - 1))
    #     if (ymin) > max_dif * (ylength - 1):
    #         ymin = round(max_dif * (ylength - 1))
    #     if (xlength - xmax) > max_dif * xlength:
    #         xmax = round((1 - max_dif) * (xlength - 1))
    #     if (xmin) > max_dif * (xlength - 1):
    #         xmin = round(max_dif * (xlength - 1))

    #     # Make sure we have enough binning in the first place
    #     if (xmax - xmin) < 20:
    #         raise Exception(
    #             "Too few bins remaining for a good PDF in X-dir (bin finer?); %i - %i = %i"
    #             % (xmax, xmin, xmax - xmin)
    #         )
    #     if (ymax - ymin) < 20:
    #         raise Exception(
    #             "Too few bins remaining for a good PDF in Y-dir (bin finer?); %i - %i = %i"
    #             % (ymax, ymin, ymax - ymin)
    #         )

    #     return xmin, xmax, ymin, ymax

    # def determine_pull(self, xmin, xmax, ymin, ymax):
    #     # is this the best?
    #     nptsy = min((ymax - ymin) // 2, 10)
    #     nptsx = min((xmax - xmin) // 2, 10)

    #     # find dx/dy for slope calculation
    #     dx_l = (self.xcenters)[: nptsx - 1] - (self.xcenters)[1:nptsx]
    #     dx_r = (self.xcenters)[-nptsx + 2 :] - (self.xcenters)[-nptsx:-2]
    #     dy_b = (self.ycenters)[: nptsy - 1] - (self.ycenters)[1:nptsy]
    #     dy_t = (self.ycenters)[-nptsy + 2 :] - (self.ycenters)[-nptsy:-2]

    #     # define left pull
    #     left_domain = self.nll[:nptsx, :]
    #     left_rows = np.column_stack(left_domain)
    #     left_pull = [-np.mean((row[:-1] - row[1:]) / dx_l) for row in left_rows]

    #     # define right pull
    #     right_domain = self.nll[-nptsx + 1 :, :]
    #     right_rows = np.column_stack(right_domain)
    #     right_pull = [np.mean((row[:-1] - row[1:]) / dx_r) for row in right_rows]

    #     # define bottom pull
    #     bottom_domain = bottom_rows = self.nll[:, :nptsy]
    #     bottom_pull = [-np.mean((row[:-1] - row[1:]) / dy_b) for row in bottom_rows]

    #     # define top pull
    #     top_domain = top_rows = self.nll[:, -nptsy + 1 :]
    #     top_pull = [np.mean((row[:-1] - row[1:]) / dy_t) for row in top_rows]

    #     return list(
    #         map(np.asarray, [left_pull, right_pull, bottom_pull, top_pull])
    #     )  # convert each list to a numpy array

    # def __call__(self, x, y):

    #     # calculate NLL at each inputted point (usually 0-4 points depending on simulation)
    #     return self.interp(x, y, grid=False)

    # def debug_plot(self, nll):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, title="NLL")
    #     im = ax.imshow(
    #         self.interp(self.xcenters, self.ycenters).T,
    #         aspect="auto",
    #         origin="lower",
    #         extent=[self.xmin, self.xmax, self.ymin, self.ymax],
    #     )
    #     plt.colorbar(im, ax=ax)
    #     ax.set_xlabel(r"cos[$\alpha$]")
    #     ax.set_ylabel("$t_{resid}$")
    #     plt.show()
    #     plt.close("all")

    #     return
