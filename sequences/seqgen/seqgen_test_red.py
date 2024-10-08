# -*- coding: utf-8 -*-
"""
Created on Tue June 11 11/06/24/ 14:00

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
    seq_output_dict = seq.waveforms_export()
    output_seq(seq_output_dict, filename, param)
    seq.plot(save=True, plot_now=False)
    return seq


def output_seq(dict, filename, param):
    """
    The interpretation from pypulseq format of sequence to the files needed to analog part of MRI

    :param dict: Dictionary of the impulse sequence pypulseq provided

    :return: files in "grad_output/" directory of every type of amplitudes and time points

    """
    '''
    Gradient
    '''
    loc_t_gx = gradient_time_convertation(param, dict['t_gx'])
    loc_t_gy = gradient_time_convertation(param, dict['t_gy'])
    loc_t_gz = gradient_time_convertation(param, dict['t_gz'])
    loc_gx = gradient_ampl_convertation(param, dict['gx'])
    loc_gy = gradient_ampl_convertation(param, dict['gy'])
    loc_gz = gradient_ampl_convertation(param, dict['gz'])
    gx_out = duplicates_delete(np.transpose([loc_t_gx, loc_gx]))
    gy_out = duplicates_delete(np.transpose([loc_t_gy, loc_gy]))
    gz_out = duplicates_delete(np.transpose([loc_t_gz, loc_gz]))
    np.savetxt('gx.txt', gx_out, fmt='%10.0f')
    np.savetxt('gy.txt', gy_out, fmt='%10.0f')
    np.savetxt('gz.txt', gz_out, fmt='%10.0f')
    '''
    Radio
    '''
    rf_raster_local = param.rf_raster_time
    rf_out = radio_ampl_convertation(dict["rf"], rf_raster=rf_raster_local)
    np.savetxt('rf_' + str(rf_raster_local) + '_raster.bin', rf_out, fmt='%10.0f')
    '''
    ADC
    '''


def radio_ampl_convertation(rf_ampl, rf_raster=1e-6):
    #TODO: sampling resize to raster different with seqgen
    out_rf_list = []
    for rf_iter in range(len(rf_ampl)):
        out_rf_list.append(rf_ampl[rf_iter].real)
        out_rf_list.append(rf_ampl[rf_iter].imag)
    return out_rf_list


def duplicates_delete(loc_list):
    new_list = [[0] * 2]
    for i in range(len(loc_list)):
        if loc_list[i][0] not in np.transpose(new_list)[0]:
            new_list.append(loc_list[i])
    return new_list


def gradient_time_convertation(param, time_sample):
    g_raster_time = param.grad_raster_time
    time_sample /= g_raster_time
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
    step_Hz_m = amplitude_max / amplitude_raster  # Hz/m step gradient
    gradient_dimless = gradient_herz / step_Hz_m * 1000
    # assert abs(any(gradient_dimless)) > 32768, 'Amplitude is higher than expected, check the rate number'
    return gradient_dimless


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


def save_param():
    output_filename = "grad_test_x"
    output_sequence = seqgen_GRAD_TEST_v2(param, output_filename)
    output_sequence.plot(save=True, plot_now=False)
    output_sequence.write(output_filename)
    file = open(output_filename + ".json", 'w')
    json.dump(param.__dict__, file, indent=4)
    file.close()


def adc_correction(blocks_number):
    """
    Helper function that rise times for correction of ADC events
    Вспомогательная функция получения времён для коррекции АЦП событий
    :return:    rise_time: float, stores in pulseq, related to exact type of gradient events
                    хранится в pulseq, связан с конкретным типом градиентного события
                fall_time: float, same as rise_time
                    аналогично rise_time
    """
    rise_time, fall_time = None, None
    is_adc_inside = False
    for j in range(blocks_number - 1):
        iterable_block = seq_input.get_block(block_index=j + 1)
        if iterable_block.adc is not None:
            is_adc_inside = True
            rise_time = iterable_block.gx.rise_time
            fall_time = iterable_block.gx.fall_time
    if not is_adc_inside:
        raise Exception("No ADC event found inside sequence")
    return rise_time, fall_time


def adc_event_edges(local_gate_adc):
    """
    Helper function that rise numbers of blocks of border  correction of ADC events
    Вспомогательная функция для получения номеров блоков границ коррекции АЦП событий
    :return:    num_begin_l:    int, number of time block when adc event starts
                                номер временного блока начала АЦП события
                num_finish_l:   int, same but ends
                                то же, но для окончания
    """
    num_begin_l = 0
    flag_begin = False
    flag_finish = False
    num_finish_l = 1
    for k in range(len(local_gate_adc) - 1):
        if local_gate_adc[k] != 0 and not flag_begin:
            num_begin_l = k
            flag_begin = True
        if local_gate_adc[k] != 0 and local_gate_adc[k + 1] == 0 and not flag_finish:
            num_finish_l = k
            flag_finish = True
    return num_begin_l, num_finish_l


def synchronization(N_samples):
    ### MAIN LOOP ###
    ### ОСНОВНОЙ ЦИКЛ###
    for i in range(N_samples):
        # delaying of RF event for time period of local delay
        # задержка RF события на период времени локальной задержки
        if RF_assintant[0] - RF_raster < time_sample[i] < RF_assintant[0] + RF_raster:
            RF_stop = int(RF_assintant[1] / time_step)
            gate_rf[i:RF_stop] = 1.0
            var = 1

        # mandatory disabling of RF gate due to ADC work same time
        # принудительное отключение RF-шлюза из-за одновременной работы АЦП
        gate_rf_2 = map(lambda x: time_sample[i] - ADC_raster < x < time_sample[i] + ADC_raster and 1 or 0,
                        seq_output_dict['t_adc'])
        if np.any(np.array(list(gate_rf_2)) > 0):
            gate_rf[i] = 0.0

        # TR switch with own delay before ADC turning
        # TR перключение с собственной задержкой перед включением АЦП
        gate_tr_1 = map(lambda x: time_sample[i] - ADC_raster < x < time_sample[i] + ADC_raster and 1 or 0,
                        seq_output_dict['t_adc'])
        if np.any(np.array(list(gate_tr_1)) > 0):
            block_delay_tr = int(local_delay_tr / time_step)
            gate_tr_switch[i - block_delay_tr:i + 1] = 0.0

        # first step of ADC gate - enabling
        # первый шак АЦП шлюза - включение
        gate_adc_1 = map(lambda x: time_sample[i] - ADC_raster < x < time_sample[i] + ADC_raster and 1 or 0,
                         seq_output_dict['t_adc'])
        if np.any(np.array(list(gate_adc_1)) > 0):
            gate_adc[i] = 1.0

    # adc correction sue to rise and fall time of gradient
    # АЦП коррекция в зависимости от времени нарастания или спада градиента
    # defining time that ADC need to be disabled during of
    # определение премени, когда АЦП необходимо отключить
    rise_time_loc, fall_time_loc = adc_correction()
    num_beg, num_fin = adc_event_edges(gate_adc)
    rise_time_tick = int(rise_time_loc / time_step)
    fall_time_tick = int(rise_time_loc / time_step)
    gate_adc[num_beg:num_beg + rise_time_tick] = 0.0
    gate_adc[num_fin - fall_time_tick:num_fin + 1] = 0.0

    gates_release = {"adc": gate_adc,
                     "rf": gate_rf,
                     "tr_switch": gate_tr_switch,
                     "gx": gate_gx,
                     "gy": gate_gy,
                     "gz": gate_gz}


if __name__ == "__main__":
    set_limits()
    save_param()
    # print(seqgen_GRAD_TEST_v2(param, "test1"))
