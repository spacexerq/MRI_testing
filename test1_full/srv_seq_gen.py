# -*- coding: utf-8 -*-
"""
Created on 05/09/2024

@author: spacexer
"""
from copy import deepcopy as dcopy
import pypulseq as pp
import numpy as np
from pulseq_fixed import sequence_fixed as puls_fix
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from datetime import datetime
import json
import asyncio

from utilities.param_constants import param_rf_GRE

rf = param_rf_GRE()

AU = 128
NA = 128
t_grad = 10e-3
delay_2 = 20e-3
grad1_pol = 1


def set_limits():
    # Задание общих аппаратных характкристик
    gamma = 42.576e6  # Hz/T    Гиромагнитное отношение водорода
    G_amp_max_mT_m = 9  # mT/m.   Максимальный градиент
    G_amp_max = G_amp_max_mT_m * 1e-3 * gamma  # Hz/m.   Максимальный градиент
    G_slew_max_T_m_s = 30  # T/m/s.  Максимальная скорость нарастания
    G_slew_max = G_slew_max_T_m_s * gamma  # Hz/m/s. Максимальная скорость нарастания
    rf_raster_time = 1e-6  # s.      Растр РЧ импульса
    grad_raster_time = 10e-6  # s.      Растр градиентов
    tau_max = G_amp_max / G_slew_max  # s.      Максимальное время нарастания градиента с учетом макс скорости нарастания
    tau_max = np.ceil(tau_max / grad_raster_time) * grad_raster_time

    N_grad = 128
    grad_step = G_amp_max / N_grad
    grad_amp = grad_step * AU

    AU_min, AU_max = 0, 128
    NA_min, NA_max = 1, "-"
    t_grad_min, t_grad_max = 1e-3, 50e-3
    delay_1_min, delay_1_max = 1e-3, 50e-3
    delay_2_min, delay_2_max = 1e-3, 50e-3

    global param
    param = SimpleNamespace()
    param.G_amp_max = G_amp_max
    param.G_slew_max = G_slew_max
    param.gamma = gamma
    param.grad_raster_time = grad_raster_time
    param.rf_raster_time = rf_raster_time

    param.grad_amp = grad_amp
    param.NA = NA
    param.t_grad = t_grad
    param.grad1_pol = grad1_pol
    param.filename = "test_seq.png"
    param.FA = 90


def save_param(path='sequences/'):
    output_filename = "test1_full"
    output_sequence = seqgen_GRAD_TEST_v2(param, output_filename)
    output_sequence.plot(save=True, plot_now=False)
    output_sequence.write(path+output_filename)
    file = open(path+output_filename + ".json", 'w')
    json.dump(param.__dict__, file, indent=4)
    file.close()


def seqgen_GRAD_TEST_v2(param, filename):
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

    t_grad = 1e-3
    delta_time = 0.5e-3
    grad1_pol = 1
    delay1 = 1e-3
    N_grad = 128
    grad_step = param.G_amp_max / N_grad
    grad_amp = grad_step
    scanner_parameters.max_grad = scanner_parameters.max_grad * 1.001
    seq = pp.Sequence(system=scanner_parameters)
    g_raster = param.grad_raster_time

    for i in range(N_grad):
        if i == 7:
            break
        if i <= 100:
            gradient_1 = pp.make_trapezoid(channel='x',
                                           flat_time=np.double(t_grad),
                                           amplitude=grad_amp * (N_grad - i - 1) * -grad1_pol,
                                           system=scanner_parameters)

            delay1 = param.grad_raster_time * np.floor(delay1 / g_raster)

            # Generate the TE delay
            delay1_grad = pp.make_delay(delay1)
            phase_shift = 0
            t_grad += delta_time
        else:
            delay1 += delta_time
            gradient_1 = pp.make_trapezoid(channel='x',
                                           flat_time=np.double(t_grad),
                                           amplitude=grad_amp * (i + 1) * grad1_pol,
                                           system=scanner_parameters)

            delay1 = param.grad_raster_time * np.floor(delay1 / g_raster)

            # Generate the TE delay
            delay1_grad = pp.make_delay(delay1)
            phase_shift = 0
        seq.add_block(gradient_1)
        seq.add_block(delay1_grad)

        exc_pulse = pp.make_sinc_pulse(flip_angle=np.radians(param.FA),
                                       duration=np.double(rf.t_ex),
                                       # freq_offset = curr_offset,
                                       phase_offset=0,
                                       time_bw_product=3.8,
                                       delay=gradient_ramp_time,
                                       system=scanner_parameters)
        seq.add_block(exc_pulse)
    #
    # grad1_pol = -1
    # delay1 = 1e-3
    # t_grad = 1e-3
    # for i in range(N_grad):
    #     if i == 7:
    #         break
    #     if i <= 100:
    #         gradient_1 = pp.make_trapezoid(channel='x',
    #                                        flat_time=np.double(t_grad),
    #                                        amplitude=grad_amp * (i + 1) * grad1_pol,
    #                                        system=scanner_parameters)
    #
    #         delay1 = param.grad_raster_time * np.floor(delay1 / g_raster)
    #
    #         # Generate the TE delay
    #         delay1_grad = pp.make_delay(delay1)
    #
    #         phase_shift = 0
    #
    #         t_grad += delta_time
    #     else:
    #         delay1 += delta_time
    #         gradient_1 = pp.make_trapezoid(channel='x',
    #                                        flat_time=np.double(t_grad),
    #                                        amplitude=grad_amp * (i + 1) * grad1_pol,
    #                                        system=scanner_parameters)
    #
    #         delay1 = param.grad_raster_time * np.floor(delay1 / g_raster)
    #
    #         # Generate the TE delay
    #         delay1_grad = pp.make_delay(delay1)
    #
    #         phase_shift = 0
    #
    #     seq.add_block(gradient_1)
    #     seq.add_block(delay1_grad)

    # grad1_pol = 1
    # delay1 = 1e-3
    # t_grad = 1e-3
    # for i in range(N_grad):
    #     gradient_1 = pp.make_trapezoid(channel='x',
    #                                    flat_time=np.double(t_grad),
    #                                    amplitude=grad_amp * (i + 1) * grad1_pol,
    #                                    system=scanner_parameters)
    #     gradient_2 = pp.make_trapezoid(channel='x',
    #                                    flat_time=np.double(t_grad),
    #                                    amplitude=grad_amp * (i + 1) * (-grad1_pol),
    #                                    system=scanner_parameters)
    #
    #     delay1 = param.grad_raster_time * np.floor(delay1 / g_raster)
    #
    #     # Generate the TE delay
    #     delay1_grad = pp.make_delay(delay1)
    #     phase_shift = 0
    #     t_grad += delta_time
    #     seq.add_block(gradient_1)
    #     seq.add_block(gradient_2)
    #     seq.add_block(delay1_grad)
    #     t_grad += delta_time

    # print(pp.check_timing.check_timing(seq))
    # seq_output_dict = seq.waveforms_export()
    # output_seq(seq_output_dict, filename, param)
    # seq.plot(save=False, plot_now=False)
    return seq


if __name__ == "__main__":
    set_limits()
    save_param()
    # print(seqgen_GRAD_TEST_v2(param, "test1"))
