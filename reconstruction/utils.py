import jax
import jax.numpy as jnp
import numpy as np

def polyform(x, *args):
    """A convenient form to fit to a cos(
    ) distribution.

        args are cher_ang,cher_off,[poly_left],[poly_right]
        poly_ lacks the offset which is constrained to be cher_off for both
        poly_ is expanded around cher_ang."""
    total = len(args)
    poly = int((total - 1) / 2)

    cher_ang, cher_off = args[:2]
    poly_left = np.concatenate([args[1: 1 + poly], [cher_off]])
    poly_right = np.concatenate([args[1 + poly:], [cher_off]])

    results = np.empty_like(x)
    mask = x <= cher_ang
    results[mask] = np.polyval(poly_left, x[mask] - cher_ang)
    mask = np.logical_not(mask)
    results[mask] = np.polyval(poly_right, x[mask] - cher_ang)

    return results

def polyform_jax(x, *args):
    """A convenient form to fit to a cos(
    ) distribution.

        args are cher_ang,cher_off,[poly_left],[poly_right]
        poly_ lacks the offset which is constrained to be cher_off for both
        poly_ is expanded around cher_ang."""
    total = len(args)
    poly = int((total - 1) / 2)

    cher_ang, cher_off = args[:2]
    
    poly_left = jnp.array([*args[1: 1 + poly], *[cher_off]])
    poly_right = jnp.array([*args[1 + poly:], *[cher_off]])

    mask = x <= cher_ang
    results = jnp.where(mask, jnp.polyval(poly_left, x - cher_ang), jnp.polyval(poly_right, x - cher_ang))

    return results