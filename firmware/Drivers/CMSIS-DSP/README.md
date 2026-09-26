# CMSIS-DSP (vendored subset)

The hybrid frontend's STFT runs on the Cortex-M55 through CMSIS-DSP's real FFT
(`arm_rfft_fast_f32`) and complex magnitude (`arm_cmplx_mag_f32`), which use the
M55's Helium vector extension when built with `ARM_MATH_HELIUM`. Only the files
those two functions need are kept here, unmodified.

- Upstream: https://github.com/ARM-software/CMSIS-DSP, tag `v1.16.2`
- License: Apache-2.0 (`LICENSE`)
- Kept: `Include/`, `PrivateInclude/`, and the ten sources under `Source/`

GCC needs `-flax-vector-conversions` for the Helium code. A host build (the
native firmware tests) defines `__GNUC_PYTHON__`, which selects CMSIS-DSP's
portable path without CMSIS-Core headers.
