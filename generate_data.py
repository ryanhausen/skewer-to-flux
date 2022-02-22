# MIT License

# Copyright (c) 2022 Computational Astrophysics Research Group

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Code to calculate the flux for a skewer, call calculate_flux on skewer data."""


import math
from typing import Tuple

import tensorflow as tf

# data constants
IDX_HI = 1
IDX_VELOCITY = 2
IDX_TEMPERATURE = 3
N = 2048
N_GHOST = int(N * 0.1)

const_f32 = lambda x: tf.constant(x, dtype=tf.float32)
# cosmology constants
COSMOLOGY_z = const_f32(2)  # redshift
COSMOLOGY_a = const_f32(1 / (COSMOLOGY_z + 1))  # scale
COSMOLOGY_M_p = const_f32(1.6726219e-24)  # mass of the proton (g)
COSMOLOGY_M_sun = const_f32(1.98847e33)  # solar mass (g)
COSMOLOGY_kpc = const_f32(1000 * 3.0857e18)  # kpc
COSMOLOGY_H0 = const_f32(67.66)
COSMOLOGY_H = const_f32(COSMOLOGY_H0 / 100)

# H is diferent in optical depth velocity calling HH
COSMOLOGY_Omega_L = const_f32(0.6889)
COSMOLOGY_Omega_M = const_f32(0.3111)
COSMOLOGY_a_dot = const_f32(
    tf.math.sqrt(COSMOLOGY_Omega_M / COSMOLOGY_a + COSMOLOGY_Omega_L * COSMOLOGY_a**2)
    * (COSMOLOGY_H0 / 1000)
)
COSMOLOGY_HH = const_f32(COSMOLOGY_a_dot / COSMOLOGY_a)
COSMOLOGY_HH_CGS = const_f32(COSMOLOGY_HH * 1e5 / COSMOLOGY_kpc)

COSMOLOGY_e_charge = const_f32(4.8032e-10)  # electron charge (cm^3/2 g^1/2 s^-1)
COSMOLOGY_M_e = const_f32(9.10938356e-28)  # electron mass (g)
COSMOLOGY_c = const_f32(2.99792458e10)  # speed of light (cm/s)
COSMOLOGY_f12 = const_f32(0.416)  # oscillator strength
COSMOLOGY_Lya_lambda = const_f32(
    1.21567e-5
)  # rest wave length of the lyman alpha transition (cm)
COSMOLOGY_Lya_sigma = const_f32(
    math.pi
    * COSMOLOGY_e_charge**2
    / COSMOLOGY_M_e
    / COSMOLOGY_c
    * COSMOLOGY_f12
    * COSMOLOGY_Lya_lambda
)

COSMOLOGY_Lbox = const_f32(50000.0)  # Box parameters (kpc/h)
COSMOLOGY_R = const_f32(COSMOLOGY_a * COSMOLOGY_Lbox / COSMOLOGY_H)
COSMOLOGY_dr = const_f32(COSMOLOGY_R / N)
COSMOLOGY_dr_cgs = const_f32(COSMOLOGY_dr * COSMOLOGY_kpc)
COSMOLOGY_r_proper = tf.constant(
    (
        tf.linspace(
            const_f32(-1.0) * const_f32(N_GHOST),
            const_f32(N) + (const_f32(N_GHOST) - const_f32(1.0)),
            N + 2 * N_GHOST,
        )
        + 0.5
    )
    * COSMOLOGY_dr,
    dtype=tf.float32,
    shape=[1, N + 2 * N_GHOST],
)
COSMOLOGY_vel_hubble = const_f32(COSMOLOGY_HH * COSMOLOGY_r_proper * const_f32(1e5))

COSMOLOGY_K_b = const_f32(1.38064852e-16)  # Boltazman constant g (cm/s)^2 K-1


def extend_array_periodic(arr: tf.Tensor) -> tf.Tensor:
    return tf.concat(
        (
            arr[:, N - N_GHOST :],  # end of the array
            arr,  # array
            arr[:, :N_GHOST],  # beginning of the array
        ),
        axis=1,
    )


def get_doppler_parameter(temperature: tf.Tensor) -> tf.Tensor:
    return tf.math.sqrt(2 * COSMOLOGY_K_b / COSMOLOGY_M_p * temperature)


def calculate_optical_depth_velocity(
    n_HI_los: tf.Tensor,
    velocity_cgs: tf.Tensor,
    temperature: tf.Tensor,
) -> tf.Tensor:
    n_HI = extend_array_periodic(const_f32(n_HI_los))
    tmp_n_HI = tf.expand_dims(n_HI, axis=1)  # [16, 1, n]

    y_l = COSMOLOGY_vel_hubble - const_f32(0.5) * COSMOLOGY_HH_CGS * COSMOLOGY_dr_cgs
    y_r = COSMOLOGY_vel_hubble + const_f32(0.5) * COSMOLOGY_HH_CGS * COSMOLOGY_dr_cgs

    # add dimensions to support broadcasting along the correct dimension
    y_l = tf.expand_dims(y_l, axis=-1)  # [16, n, 1]
    y_r = tf.expand_dims(y_r, axis=-1)  # [16, n, 1]

    velocity = COSMOLOGY_vel_hubble + extend_array_periodic(velocity_cgs)
    tmp_velocity = tf.cast(
        tf.expand_dims(velocity, axis=1), dtype=tf.float32
    )  # [16, 1, n]
    b_all = get_doppler_parameter(extend_array_periodic(temperature))
    tmp_b_all = tf.cast(tf.expand_dims(b_all, axis=1), dtype=tf.float32)  # [16, 1, n]

    y_l = y_l - tmp_velocity  # [16, n, n]
    y_r = y_r - tmp_velocity  # [16, n, n]
    y_l = y_l / tmp_b_all  # [16, n, n]
    y_r = y_r / tmp_b_all  # [16, n, n]

    tau_los = tf.math.reduce_sum(
        tmp_n_HI * (tf.math.erf(y_r) - tf.math.erf(y_l)), axis=-1
    )  # [16, n]
    tau_los = COSMOLOGY_Lya_sigma / COSMOLOGY_HH_CGS * tau_los / 2

    tau_los = tf.slice(tau_los, begin=[0, N_GHOST], size=[tau_los.shape[0], N])

    return tau_los


def calculate_optical_depth(data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    hi = data[:, :, IDX_HI]  # [b, 2048]
    velocity = data[:, :, IDX_VELOCITY]  # [b, 2048]
    temperature = data[:, :, IDX_TEMPERATURE]  # [b, 2048]

    dens_HI_los = hi / COSMOLOGY_a**3  # [b, 2048]

    # cgs_scale = tf.cast(
    #     tf.cast(COSMOLOGY_M_sun, dtype=tf.float64) / tf.cast(COSMOLOGY_kpc, dtype=tf.float64)**3,
    #     dtype=tf.float32
    # ) * COSMOLOGY_H**2
    # this version doesn't require type promotion
    cgs_scale = (
        COSMOLOGY_M_sun
        / COSMOLOGY_kpc
        / COSMOLOGY_kpc
        / COSMOLOGY_kpc
        * COSMOLOGY_H**2
    )

    dens_HI_los_cgs = dens_HI_los * cgs_scale  # [b, 2048]
    n_HI_los = dens_HI_los_cgs / COSMOLOGY_M_p  # [b, 2048]

    velocity_cgs = velocity * const_f32(1e5)  # [b, 2048]

    tau = calculate_optical_depth_velocity(n_HI_los, velocity_cgs, temperature)

    return tau


def calculate_flux(skewers: tf.Tensor) -> tf.Tensor:
    """Calculates the flux for a the passed in skewers.

    Inputs:
        skewers (tf.Tensor): the input skewers should be [batch, 2048, 4]

    Returns:
        a skewer of flux values, should have shape [batch, 2048]
    """

    tau = calculate_optical_depth(skewers)
    return tf.math.exp(-tau)


if __name__=="__main__":
    import os
    import numpy as np
    from tqdm import tqdm

    if os.path.exists("data_ys.npy"):
        ys = np.load("data_ys.npy").astype(np.float32)

        xs = np.zeros(list(ys.shape[:-1]) + [1], dtype=np.float32)

        for i in tqdm(range(0, ys.shape[0], 10)):
            xs[i:i+10, ...] = np.expand_dims(calculate_flux(ys[i:i+10, ...]), axis=-1)

        np.save("data_xs.npy", xs)
        np.save("data_ys.npy", ys[..., 1:])
    else:
        print("data_ys.npy does not exist. Run 'python download_labels.py'")