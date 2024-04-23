# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:43:09 2024

@author: zilya
"""
from copy import deepcopy as dcopy
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
    output_seq(seq_output_dict, filename, param)
    return seq


def output_seq(dict, filename, param):
    """
    The interpretation from pypulseq format of sequence to the files needed to analog part of MRI

    :param dict: Dictionary of the impulse sequence pypulseq provided

    :return: files in "grad_output/" directory of every type of amplitudes and time points

    """
    loc_t_gx = gradient_time_convertation(param, dict['t_gx'])
    loc_t_gy = gradient_time_convertation(param, dict['t_gy'])
    loc_t_gz = gradient_time_convertation(param, dict['t_gz'])
    loc_gx = gradient_ampl_convertation(param, dict['gx'])
    loc_gy = gradient_ampl_convertation(param, dict['gy'])
    loc_gz = gradient_ampl_convertation(param, dict['gz'])
    gx_out = [loc_gx, loc_t_gx]
    gy_out = [loc_gy, loc_t_gy]
    gz_out = [loc_gz, loc_t_gz]
    out_name = "grad_output/" + filename + "_"
    np.savetxt(out_name + 'gx.txt', np.transpose(gx_out), fmt='%10.0f')
    np.savetxt(out_name + 'gy.txt', np.transpose(gy_out), fmt='%10.0f')
    np.savetxt(out_name + 'gz.txt', np.transpose(gz_out), fmt='%10.0f')


def gradient_time_convertation(param, time_sample):
    g_raster_time = param.grad_raster_time
    time_sample/=g_raster_time
    return time_sample


def gradient_ampl_convertation(param, gradient_herz):
    """
    Helper function that convert amplitudes to dimensionless format for machine
    1 bit for sign, 15 bits of numbers

    :param gradient_herz: 2D array of amplitude and time points in Hz/m

    :return: gradient_dimless: 2D array of dimensionless points

    """
    # amplitude raster is 32768
    # maximum grad = 10 mT/m
    # artificial gap is 1 mT/m so 9 mT/m is now should be split in parts
    amplitude_max = param.G_amp_max
    amplitude_raster = 32767
    step_Hz_m = amplitude_max/amplitude_raster  # Hz/m step gradient
    gradient_dimless = gradient_herz/step_Hz_m*1000
    # assert abs(any(gradient_dimless)) > 32768, 'Amplitude is higher than expected, check the rate number'
    return gradient_dimless
