import resource
import subprocess as sp
from geometry.theia import det_radius

import numpy as np
import numexpr as ne

# Mask function for getting wavelengths
def get_mask(flags, test, none_of=None):
    if none_of is not None:
        has_test = np.bitwise_and(flags, test) == test
        has_none_of = np.bitwise_and(flags, none_of) == 0
        return np.logical_and(has_test, has_none_of)
    else:
        return np.bitwise_and(flags, test) == test
    
def count_test(flags, test, none_of=None):
    return np.count_nonzero(get_mask(flags, test, none_of))


def get_refractive_index(target, photocathode, verbose=False):

    if target is not None:
        refractive_index = [b for _, b in target.refractive_index]
    else:
        raise Exception("Target is of type None")

    if photocathode is not None:
        wvl, prob = photocathode["detect"].T
    else:
        raise Exception("Photocathode is of type None")

    qe_interp = np.interp(np.linspace(200, 1000, len(refractive_index)), wvl, prob)
    refract = np.dot(qe_interp, refractive_index) / np.sum(qe_interp)

    if verbose:
        print("[refract] calculated refractive index", refract)
    return refract


def wv2rgb(wavelength, gamma=0.8):
    """
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Taken from
    https://gist.github.com/error454/65d7f392e1acd4a782fc
    """

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    # R *= 255
    # G *= 255
    # B *= 255
    return (R, G, B)


def get_gpu_memory():

    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[1]


def get_cpu_memory():
    return int(resource.getrusage(resource.RUSAGE_SELF)[2] / 1024)


refractive_index_long = 1.4876301945708197
refractive_index_short = 1.5069515555735635

# fmt: off
tresid_long_bins = [ -5.388792562475686, -2.683692714342655, -2.279271185144758, -2.0212700995520554, -1.8152032923996784, -1.6539396430965674, -1.5082947325694125, -1.381839024566434, -1.2546549264411297, -1.1439002186582898, -1.040693361786527, -0.9489150748439573, -0.8614354059348613, -0.7727248666574376, -0.684263633714042, -0.5946276234778622, -0.5151538614020638, -0.43880084767960936, -0.3605979770917912, -0.2834344347474296, -0.20821460477328912, -0.13712599707965042, -0.05549965168020345, 0.02754479445557843, 0.10617690483913639, 0.18433534716766076, 0.27158827587422935, 0.3623305411588984, 0.45150413770883724, 0.552350517318788, 0.6469758082370785, 0.7449521159309542, 0.8597848430002217, 0.9743557834808827, 1.0962315458727705, 1.2276808925554938, 1.3803511113348728, 1.5410884395129336, 1.7502208302209314, 2.0053236935230667, 2.301690531827049, 2.6401652545067735, 3.0235129165214683, 3.451679149214123, 3.843692556620843, 4.286126639641537, 4.756455320773504, 5.304580307767003, 5.821679509892853, 6.389770210045351, 7.0464899915934245, 7.720027045481996, 8.425511528021234, 9.180110472184552, 9.996173076501222, 10.835759521628367, 11.844424979652501, 12.91712545608353, 14.052969467129603, 15.258454386562663, 16.53940832652305, 17.90616963044775, 19.506025674765308, 21.289679127165343, 23.053021024241882, 24.749916648783827, 26.903272022477655, 29.126787811357076, 31.368369214492205, 33.94684849771874, 36.52038713486455, 39.251634245318584, 42.22319873749165, 45.12050195192977, 48.31113329624735, 51.63272382263968, 55.51162367489823, 59.81714151472213, 63.77578801225864, 68.14783476300752, 72.63042099716124, 77.53662840062161, 83.4090955101109, 88.98188343757108, 95.67380171666571, 102.41550397386088, 109.46636282337643, 117.86769040793017, 126.99938226624568, 136.9220072251584, 148.4403733140207, 160.74215829377064, 175.27801317238993, 191.92136381481583, 209.46480312860234, 231.71194282063226, 258.7605058767562, 293.49218051018113, 345.7248549136792, 440.0619892638366, 1425.2157411584549, ]
# fmt: on

""" 
Detection efficiencies of each photon type on each detector
"""
long_cherenkov = 0.001496400179264184
long_scintillation = 3.6124048488192975e-05
short_cherenkov = 0.03766302786216984
short_scintillation = 0.09743377330036686


def sph2cart(theta, phi, ceval=ne.evaluate):
    """
    Convert spherical coordinates to cartesian coordinates
    """
    
    x = ceval("cos(phi) * sin(theta)")
    y = ceval("sin(phi) * sin(theta)")
    z = ceval("cos(theta)")
    return np.asarray(x, y, z)

def random_three_vector():
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.asarray([x, y, z])

def random_pos(height, radius, *, buffer=2000):
    # make sure units are in mm!
    l, w, h = 2 * (radius - buffer), 2 * (radius - buffer), (height - 2 * buffer) / 2
    x, y, z = (
        np.random.uniform(-l, l),
        np.random.uniform(-w, w),
        np.random.uniform(-h, h),
    )
    random_radius = np.sqrt(np.square(x) + np.square(y))
    if random_radius > radius - buffer:
        return random_pos(height, radius, buffer=buffer)
    else:
        return np.asarray([x, y, z])
    
    
def is_in_cylinder(x, radius, height, buffer=2000):
    return np.sqrt(x[0]**2 + x[1]**2) < radius-buffer and np.abs(x[2]) < (height-2*buffer)/2

THEIA_RADIUS_50KT = det_radius(50.0)
THEIA_HEIGHT_50KT = 2*THEIA_RADIUS_50KT


def dist_to_wall(x, dvec, radius=THEIA_RADIUS_50KT, height=THEIA_HEIGHT_50KT, buffer=0):
    n = 0
    while is_in_cylinder(x + n*dvec, radius, height, buffer):
        n += 1

    ns = np.linspace(n-1, n, 1000)
    for n in ns:
        if not is_in_cylinder(x + n*dvec, radius, height, buffer):
            return np.sqrt(np.sum(np.square(n*dvec), axis=0))
