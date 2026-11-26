# Cross-dataset detail (zero-shot scores)

Made by `analyze_cross_detail.py` from the saved per-image files. AUC and detection rate at 5% false alarms with bootstrap 95% intervals (2000 resamples within class, seed 0). The 5% false-alarm threshold is set on the same set's real images, so that column is an oracle operating point, not zero-shot.

| file | n real | n manip. | AUC [95% CI] | detection @5% FA [95% CI] |
|---|---|---|---|---|
| cross_allreals_dmaer | 1000 | 1000 | 0.665 [0.640, 0.687] | 0.115 [0.088, 0.156] |
| cross_dmaer | 494 | 1000 | 0.780 [0.753, 0.803] | 0.272 [0.219, 0.318] |
| cross_fmow | 60 | 60 | 0.450 [0.350, 0.556] | 0.067 [0.017, 0.133] |
| cross_frozen_dmaer | 494 | 1000 | 0.759 [0.730, 0.786] | 0.159 [0.101, 0.219] |
| cross_frozen_fmow | 60 | 60 | 0.359 [0.266, 0.460] | 0.017 [0.000, 0.100] |
| cross_frozen_gdd_crossdomain | 1000 | 1000 | 0.550 [0.525, 0.575] | 0.089 [0.066, 0.116] |
| cross_frozen_gdd_crossgenerator | 1000 | 1000 | 0.541 [0.515, 0.567] | 0.080 [0.049, 0.116] |
| cross_frozen_gdd_indomain | 750 | 750 | 0.515 [0.485, 0.548] | 0.193 [0.159, 0.233] |
| cross_frozen_gdd_localization | 500 | 500 | 0.505 [0.469, 0.542] | 0.036 [0.014, 0.060] |
| cross_frozen_rsfake | 5000 | 5000 | 0.543 [0.531, 0.554] | 0.051 [0.043, 0.063] |
| cross_frozen_uw | 4032 | 4032 | 0.716 [0.705, 0.726] | 0.069 [0.049, 0.091] |
| cross_gdd_crossdomain | 1000 | 1000 | 0.706 [0.684, 0.728] | 0.169 [0.144, 0.208] |
| cross_gdd_crossgenerator | 1000 | 1000 | 0.648 [0.625, 0.673] | 0.120 [0.092, 0.146] |
| cross_gdd_indomain | 750 | 750 | 0.536 [0.507, 0.565] | 0.120 [0.097, 0.153] |
| cross_gdd_localization | 500 | 500 | 0.568 [0.532, 0.603] | 0.114 [0.076, 0.150] |
| cross_rsfake | 5000 | 5000 | 0.584 [0.573, 0.595] | 0.123 [0.111, 0.139] |
| cross_uw | 4032 | 4032 | 0.500 [0.487, 0.513] | 0.044 [0.037, 0.052] |
| cross_web_allreals_dmaer | 1000 | 1000 | 0.657 [0.633, 0.679] | 0.201 [0.170, 0.233] |
| cross_web_dmaer | 494 | 1000 | 0.738 [0.713, 0.763] | 0.317 [0.264, 0.378] |
| cross_web_fmow | 60 | 60 | 0.432 [0.331, 0.536] | 0.083 [0.000, 0.167] |
| cross_web_gdd_crossdomain | 1000 | 1000 | 0.634 [0.610, 0.658] | 0.170 [0.145, 0.212] |
| cross_web_gdd_crossgenerator | 1000 | 1000 | 0.596 [0.571, 0.621] | 0.135 [0.111, 0.175] |
| cross_web_gdd_indomain | 750 | 750 | 0.481 [0.450, 0.510] | 0.129 [0.096, 0.163] |
| cross_web_gdd_localization | 500 | 500 | 0.560 [0.525, 0.596] | 0.132 [0.094, 0.168] |
| cross_web_rsfake | 5000 | 5000 | 0.668 [0.657, 0.678] | 0.156 [0.142, 0.170] |
| cross_web_uw | 4032 | 4032 | 0.527 [0.514, 0.539] | 0.041 [0.034, 0.048] |

## Per manipulated group

### cross_allreals_dmaer

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| validation/FAKE | 1000 | 0.665 | 0.112 | 0.115 |

### cross_dmaer

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| validation/FAKE | 1000 | 0.780 | 0.112 | 0.272 |

### cross_fmow

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| inpaint | 20 | 0.450 | 0.000 | 0.000 |
| object-splice | 20 | 0.361 | 0.000 | 0.000 |
| simple-splice | 20 | 0.540 | 0.000 | 0.200 |

### cross_frozen_dmaer

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| validation/FAKE | 1000 | 0.759 | 0.372 | 0.159 |

### cross_frozen_fmow

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| inpaint | 20 | 0.384 | 0.000 | 0.000 |
| object-splice | 20 | 0.277 | 0.000 | 0.000 |
| simple-splice | 20 | 0.417 | 0.150 | 0.050 |

### cross_frozen_gdd_crossdomain

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/geosynth | 334 | 0.638 | 0.246 | 0.093 |
| gen/rspaint | 333 | 0.746 | 0.408 | 0.174 |
| gen/sd35medium | 333 | 0.266 | 0.018 | 0.000 |

### cross_frozen_gdd_crossgenerator

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/geosynth | 334 | 0.629 | 0.246 | 0.081 |
| gen/rspaint | 333 | 0.739 | 0.408 | 0.159 |
| gen/sd35medium | 333 | 0.254 | 0.018 | 0.000 |

### cross_frozen_gdd_indomain

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/diffusionsat | 150 | 0.675 | 0.220 | 0.040 |
| gen/flux2dev | 150 | 0.152 | 0.000 | 0.000 |
| gen/flux2kleinbase4b | 150 | 0.039 | 0.000 | 0.000 |
| gen/neds | 150 | 0.954 | 0.907 | 0.747 |
| gen/r3gan | 150 | 0.755 | 0.460 | 0.180 |

### cross_frozen_gdd_localization

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| manip/fully-regenerated | 84 | 0.725 | 0.393 | 0.083 |
| manip/inpaint-splice | 84 | 0.471 | 0.048 | 0.012 |
| manip/object-copy-move | 83 | 0.450 | 0.084 | 0.048 |
| manip/object-splice | 83 | 0.421 | 0.072 | 0.036 |
| manip/simple-copy-move | 83 | 0.471 | 0.157 | 0.036 |
| manip/simple-splice | 83 | 0.492 | 0.096 | 0.000 |

### cross_frozen_rsfake

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| generated_RSSD_768 | 500 | 0.147 | 0.002 | 0.000 |
| generated_SDFRS | 500 | 0.743 | 0.480 | 0.070 |
| generated_crsdiff | 500 | 0.683 | 0.462 | 0.120 |
| generated_diffusion_sat | 500 | 0.519 | 0.156 | 0.010 |
| generated_diffusion_sat_256 | 500 | 0.762 | 0.546 | 0.102 |
| generated_geosynth | 500 | 0.322 | 0.026 | 0.000 |
| generated_geosynth_canny | 500 | 0.621 | 0.314 | 0.048 |
| generated_geosynth_sam | 500 | 0.436 | 0.076 | 0.000 |
| generated_mapsat | 500 | 0.520 | 0.188 | 0.010 |
| generated_rsinpaint | 500 | 0.672 | 0.446 | 0.152 |

### cross_frozen_uw

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| fake/Beijing_style | 2016 | 0.699 | 0.242 | 0.051 |
| fake/Seattle_style | 2016 | 0.732 | 0.430 | 0.088 |

### cross_gdd_crossdomain

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/geosynth | 334 | 0.770 | 0.006 | 0.066 |
| gen/rspaint | 333 | 0.877 | 0.222 | 0.435 |
| gen/sd35medium | 333 | 0.471 | 0.000 | 0.006 |

### cross_gdd_crossgenerator

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/geosynth | 334 | 0.709 | 0.006 | 0.015 |
| gen/rspaint | 333 | 0.836 | 0.222 | 0.342 |
| gen/sd35medium | 333 | 0.400 | 0.000 | 0.003 |

### cross_gdd_indomain

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/diffusionsat | 150 | 0.599 | 0.007 | 0.013 |
| gen/flux2dev | 150 | 0.179 | 0.000 | 0.000 |
| gen/flux2kleinbase4b | 150 | 0.196 | 0.000 | 0.000 |
| gen/neds | 150 | 0.894 | 0.340 | 0.447 |
| gen/r3gan | 150 | 0.812 | 0.080 | 0.140 |

### cross_gdd_localization

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| manip/fully-regenerated | 84 | 0.852 | 0.214 | 0.381 |
| manip/inpaint-splice | 84 | 0.571 | 0.095 | 0.179 |
| manip/object-copy-move | 83 | 0.500 | 0.024 | 0.036 |
| manip/object-splice | 83 | 0.442 | 0.024 | 0.036 |
| manip/simple-copy-move | 83 | 0.486 | 0.000 | 0.024 |
| manip/simple-splice | 83 | 0.554 | 0.000 | 0.024 |

### cross_rsfake

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| generated_RSSD_768 | 500 | 0.268 | 0.002 | 0.002 |
| generated_SDFRS | 500 | 0.816 | 0.280 | 0.310 |
| generated_crsdiff | 500 | 0.764 | 0.150 | 0.178 |
| generated_diffusion_sat | 500 | 0.574 | 0.038 | 0.052 |
| generated_diffusion_sat_256 | 500 | 0.804 | 0.224 | 0.254 |
| generated_geosynth | 500 | 0.387 | 0.006 | 0.012 |
| generated_geosynth_canny | 500 | 0.512 | 0.042 | 0.050 |
| generated_geosynth_sam | 500 | 0.407 | 0.016 | 0.020 |
| generated_mapsat | 500 | 0.527 | 0.058 | 0.068 |
| generated_rsinpaint | 500 | 0.778 | 0.264 | 0.284 |

### cross_uw

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| fake/Beijing_style | 2016 | 0.395 | 0.007 | 0.014 |
| fake/Seattle_style | 2016 | 0.605 | 0.058 | 0.075 |

### cross_web_allreals_dmaer

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| validation/FAKE | 1000 | 0.657 | 0.184 | 0.201 |

### cross_web_dmaer

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| validation/FAKE | 1000 | 0.738 | 0.184 | 0.317 |

### cross_web_fmow

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| inpaint | 20 | 0.427 | 0.000 | 0.100 |
| object-splice | 20 | 0.323 | 0.000 | 0.100 |
| simple-splice | 20 | 0.548 | 0.000 | 0.050 |

### cross_web_gdd_crossdomain

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/geosynth | 334 | 0.709 | 0.024 | 0.054 |
| gen/rspaint | 333 | 0.893 | 0.195 | 0.453 |
| gen/sd35medium | 333 | 0.301 | 0.000 | 0.003 |

### cross_web_gdd_crossgenerator

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/geosynth | 334 | 0.660 | 0.024 | 0.042 |
| gen/rspaint | 333 | 0.861 | 0.195 | 0.360 |
| gen/sd35medium | 333 | 0.268 | 0.000 | 0.003 |

### cross_web_gdd_indomain

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| gen/diffusionsat | 150 | 0.552 | 0.000 | 0.013 |
| gen/flux2dev | 150 | 0.147 | 0.000 | 0.007 |
| gen/flux2kleinbase4b | 150 | 0.059 | 0.000 | 0.000 |
| gen/neds | 150 | 0.806 | 0.160 | 0.300 |
| gen/r3gan | 150 | 0.840 | 0.073 | 0.327 |

### cross_web_gdd_localization

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| manip/fully-regenerated | 84 | 0.849 | 0.179 | 0.333 |
| manip/inpaint-splice | 84 | 0.604 | 0.083 | 0.214 |
| manip/object-copy-move | 83 | 0.439 | 0.000 | 0.012 |
| manip/object-splice | 83 | 0.502 | 0.036 | 0.120 |
| manip/simple-copy-move | 83 | 0.467 | 0.012 | 0.072 |
| manip/simple-splice | 83 | 0.496 | 0.012 | 0.036 |

### cross_web_rsfake

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| generated_RSSD_768 | 500 | 0.482 | 0.002 | 0.010 |
| generated_SDFRS | 500 | 0.856 | 0.222 | 0.312 |
| generated_crsdiff | 500 | 0.793 | 0.156 | 0.260 |
| generated_diffusion_sat | 500 | 0.732 | 0.042 | 0.072 |
| generated_diffusion_sat_256 | 500 | 0.890 | 0.312 | 0.418 |
| generated_geosynth | 500 | 0.439 | 0.002 | 0.010 |
| generated_geosynth_canny | 500 | 0.577 | 0.046 | 0.064 |
| generated_geosynth_sam | 500 | 0.451 | 0.002 | 0.004 |
| generated_mapsat | 500 | 0.621 | 0.088 | 0.134 |
| generated_rsinpaint | 500 | 0.837 | 0.174 | 0.276 |

### cross_web_uw

| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |
|---|---|---|---|---|
| fake/Beijing_style | 2016 | 0.527 | 0.041 | 0.046 |
| fake/Seattle_style | 2016 | 0.526 | 0.031 | 0.036 |
