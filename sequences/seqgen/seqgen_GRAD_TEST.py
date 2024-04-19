# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:43:09 2024

@author: zilya
"""

import pypulseq as pp
import numpy as np
from pulseq_fixed import sequence_fixed as puls_fix

def seqgen_GRAD_TEST(param, filename):
    # Read scanner parameters from the params structure
    scanner_parameters = pp.Opts(max_grad=param.G_amp_max, grad_unit='Hz/m',
                                 max_slew=param.G_slew_max, slew_unit='Hz/m/s',
                                 grad_raster_time=param.grad_raster_time,
                                 rf_raster_time=param.rf_raster_time,
                                 block_duration_raster=max(param.grad_raster_time, param.rf_raster_time))

    # For all modules
    gradient_ramp_time = param.grad_raster_time * np.ceil(
        (scanner_parameters.max_grad / scanner_parameters.max_slew) / param.grad_raster_time)

    # Calculate slice-selection gradient
    gradient_1 = pp.make_trapezoid(channel='x',
                                   flat_time=np.double(param.t_grad),
                                   amplitude=param.grad_amp * param.grad1_pol,
                                   system=scanner_parameters)

    gradient_2 = pp.make_trapezoid(channel='x',
                                   flat_time=np.double(param.t_grad),
                                   amplitude=param.grad_amp * param.grad2_pol,
                                   system=scanner_parameters)

    # Calculate the TE pause time
    delay1 = param.delay_1
    delay1 = param.grad_raster_time * np.floor(delay1 / param.grad_raster_time)

    delay2 = param.delay_2
    delay2 = param.grad_raster_time * np.floor(delay2 / param.grad_raster_time)

    # Generate the TE delay
    delay1 = pp.make_delay(delay1)
    delay2 = pp.make_delay(delay2)

    scanner_parameters.max_grad = scanner_parameters.max_grad * 1.001
    seq = pp.Sequence(system=scanner_parameters)
    phase_shift = 0
    # Dummy scans
    for i in range(param.NA):
        seq.add_block(gradient_1)
        seq.add_block(delay1)
        seq.add_block(gradient_2)
        seq.add_block(delay2)
    # print(pp.check_timing.check_timing(seq))
    seq_output_dict = seq.waveforms_export(time_range=(0, 3))
    output_seq(seq_output_dict, filename)
    return seq


def output_seq(dict, filename):
    """
    The interpretation from pypulseq format of sequence to the files needed to analog part of MRI

    :param dict: Dictionary of the impulse sequence pypulseq provided

    :return: files in "data_output_seq/" directory of every type of amplitudes and time points

    """
    loc_t_gx = dict['t_gx']
    loc_t_gy = dict['t_gy']
    loc_t_gz = dict['t_gz']
    loc_gx = dict['gx']
    loc_gy = dict['gy']
    loc_gz = dict['gz']
    out_name = "grad_output/"+filename+"_"
    with open(out_name+'t_gx.txt', 'w') as f:
        data = str(tuple(loc_t_gx))
        f.write(data)
    with open(out_name+'t_gy.txt', 'w') as f:
        data = str(tuple(loc_t_gy))
        f.write(data)
    with open(out_name+'t_gz.txt', 'w') as f:
        data = str(tuple(loc_t_gz))
        f.write(data)
    with open(out_name+'gx.txt', 'w') as f:
        data = str(tuple(loc_gx))
        f.write(data)
    with open(out_name+'gy.txt', 'w') as f:
        data = str(tuple(loc_gy))
        f.write(data)
    with open(out_name+'gz.txt', 'w') as f:
        data = str(tuple(loc_gz))
        f.write(data)
