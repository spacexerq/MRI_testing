# -*- coding: utf-8 -*-
"""
Created on 05/09/2024

@author: spacexer
"""
from numba import njit, prange


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
from yattag import Doc, indent

from utilities.param_constants import param_rf_GRE

'''
integration from srv_seq_gen
'''

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
    '''
    only for srv_interp
    '''
    return param


def save_param(path='sequences/'):
    output_filename = "test1_full"
    output_sequence = seqgen_GRAD_TEST_v2(param, output_filename)
    output_sequence.plot(save=True, plot_now=False)
    output_sequence.write(path + output_filename)
    file = open(path + output_filename + ".json", 'w')
    json.dump(param.__dict__, file, indent=4)
    file.close()
    '''
    only for srv_interp
    '''
    return output_sequence


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
        adc_module = pp.make_adc(num_samples=1, duration=1e-5, delay=gradient_ramp_time,
                                 system=scanner_parameters)
        seq.add_block(adc_module)
    return seq


''''''''''''''''''''''''


def seq_file_input(seq_file_name="empty.seq"):
    seq_input = pp.Sequence()
    seq_input.read(file_path=seq_file_name)
    seq_output_dict = seq_input.waveforms_export()
    return seq_input, seq_output_dict


def output_seq(dict, param, path='test1/'):
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
    np.savetxt(path + 'gx.txt', gx_out, fmt='%10.0f')
    np.savetxt(path + 'gy.txt', gy_out, fmt='%10.0f')
    np.savetxt(path + 'gz.txt', gz_out, fmt='%10.0f')
    '''
    Radio
    '''
    rf_raster_local = param['rf_raster_time']
    rf_out = radio_ampl_convertation(dict["rf"], rf_raster=rf_raster_local)
    np.savetxt(path + 'rf_' + str(rf_raster_local) + '_raster.bin', rf_out, fmt='%10.0f')


def radio_ampl_convertation(rf_ampl, rf_raster=1e-6):
    #TODO: sampling resize to raster different with seqgen
    out_rf_list = []
    rf_ampl_raster = 127
    rf_ampl_maximum = np.abs(max(rf_ampl))
    proportional_cf_rf = rf_ampl_raster/rf_ampl_maximum
    for rf_iter in range(len(rf_ampl)):
        out_rf_list.append(round(rf_ampl[rf_iter].real*proportional_cf_rf))
        out_rf_list.append(round(rf_ampl[rf_iter].imag*proportional_cf_rf))
    return out_rf_list


def duplicates_delete(loc_list):
    new_list = [[0] * 2]
    for i in range(len(loc_list)):
        if loc_list[i][0] not in np.transpose(new_list)[0]:
            new_list.append(loc_list[i])
    return new_list


def gradient_time_convertation(param_loc, time_sample):
    g_raster_time = param_loc['grad_raster_time']
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
    amplitude_max = param['G_amp_max']
    amplitude_raster = 32767
    step_Hz_m = amplitude_max / amplitude_raster  # Hz/m step gradient
    gradient_dimless = gradient_herz / step_Hz_m * 1000
    # assert abs(any(gradient_dimless)) > 32768, 'Amplitude is higher than expected, check the rate number'
    return gradient_dimless

def adc_correction(blocks_number_loc, seq_input_loc):
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
    for j in range(blocks_number_loc - 1):
        iterable_block = seq_input_loc.get_block(block_index=j + 1)
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


def synchronization(N_samples, RF_assistant, seq_dict_local, path='test1'):
    ### MAIN LOOP ###
    ### ОСНОВНОЙ ЦИКЛ###

    for i in range(N_samples):
        print(str(i)+"/"+str(N_samples))
        # delaying of RF event for time period of local delay
        # задержка RF события на период времени локальной задержки
        if RF_assistant[0] - RF_raster < time_sample[i] < RF_assistant[0] + RF_raster:
            RF_stop = int(RF_assistant[1] / time_step)
            gate_rf[i:RF_stop] = 1.0
            var = 1

        # mandatory disabling of RF gate due to ADC work same time
        # принудительное отключение RF-шлюза из-за одновременной работы АЦП
        gate_rf_2 = map(lambda x: time_sample[i] - ADC_raster < x < time_sample[i] + ADC_raster and 1 or 0,
                        seq_dict_local['t_adc'])
        if np.any(np.array(list(gate_rf_2)) > 0):
            gate_rf[i] = 0.0

        # TR switch with own delay before ADC turning
        # TR перключение с собственной задержкой перед включением АЦП
        gate_tr_1 = map(lambda x: time_sample[i] - ADC_raster < x < time_sample[i] + ADC_raster and 1 or 0,
                        seq_dict_local['t_adc'])
        if np.any(np.array(list(gate_tr_1)) > 0):
            block_delay_tr = int(local_delay_tr / time_step)
            gate_tr_switch[i - block_delay_tr:i + 1] = 0.0

        # first step of ADC gate - enabling
        # первый шак АЦП шлюза - включение
        gate_adc_1 = map(lambda x: time_sample[i] - ADC_raster < x < time_sample[i] + ADC_raster and 1 or 0,
                         seq_dict_local['t_adc'])
        if np.any(np.array(list(gate_adc_1)) > 0):
            gate_adc[i] = 1.0

    # adc correction sue to rise and fall time of gradient
    # АЦП коррекция в зависимости от времени нарастания или спада градиента
    # defining time that ADC need to be disabled during of
    # определение премени, когда АЦП необходимо отключить
    # rise_time_loc, fall_time_loc = adc_correction(blocks_number, SEQ_INPUT)
    '''
    test1 swap
    '''
    rise_time_loc, fall_time_loc = 0, 0
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
    # print(gates_release)
    sync_output(gates_release, sync_raster=SYNC_STEP)


def sync_output(sync_dict, sync_raster=1e-6, path='test1/'):
    RF_value = []
    ADC_value = []
    GX_value = []
    GY_value = []
    GZ_value = []
    SW_value = []
    count_values = []
    flag_writing = True
    time_block_begin_counter = 0
    #TODO: testing all possibilities
    for sync_iter in range(len(sync_dict['gx']) - 1):
        if flag_writing:
            RF_value.append(sync_dict['rf'][sync_iter])
            ADC_value.append(sync_dict['adc'][sync_iter])
            GX_value.append(sync_dict['gx'][sync_iter])
            GY_value.append(sync_dict['gy'][sync_iter])
            GZ_value.append(sync_dict['gz'][sync_iter])
            SW_value.append(sync_dict['tr_switch'][sync_iter])
            flag_writing = False
        if any([
            sync_dict['rf'][sync_iter] != sync_dict['rf'][sync_iter + 1],
            sync_dict['adc'][sync_iter] != sync_dict['adc'][sync_iter + 1],
            sync_dict['gx'][sync_iter] != sync_dict['gx'][sync_iter + 1],
            sync_dict['gy'][sync_iter] != sync_dict['gy'][sync_iter + 1],
            sync_dict['gz'][sync_iter] != sync_dict['gz'][sync_iter + 1],
            sync_dict['tr_switch'][sync_iter] != sync_dict['tr_switch'][sync_iter + 1]
        ]):
            flag_writing = True
            time_val = sync_iter + 1 - time_block_begin_counter
            time_block_begin_counter = sync_iter + 1
            count_values.append(time_val)
    time_val = len(sync_dict['gx']) - time_block_begin_counter
    count_values.append(time_val)

    doc, tag, text = Doc().tagtext()

    number_of_blocks = len(RF_value)
    with tag('root'):
        with tag('ParamCount'):
            text(number_of_blocks)
        with tag('RF'):
            for RF_iter in range(number_of_blocks):
                with tag('RF' + str(RF_iter + 1)):
                    text(RF_value[RF_iter])
        with tag('SW'):
            for SW_iter in range(number_of_blocks):
                with tag('SW' + str(SW_iter + 1)):
                    text(SW_value[SW_iter])
        with tag('ADC'):
            for ADC_iter in range(number_of_blocks):
                with tag('ADC' + str(ADC_iter + 1)):
                    text(ADC_value[ADC_iter])
        with tag('GX'):
            for GX_iter in range(number_of_blocks):
                with tag('GX' + str(GX_iter + 1)):
                    text(GX_value[GX_iter])
        with tag('GY'):
            for GY_iter in range(number_of_blocks):
                with tag('GY' + str(GY_iter + 1)):
                    text(GY_value[GY_iter])
        with tag('GZ'):
            for GZ_iter in range(number_of_blocks):
                with tag('GZ' + str(GZ_iter + 1)):
                    text(GZ_value[GZ_iter])
        with tag('CL'):
            for CL_iter in range(number_of_blocks):
                with tag('CL' + str(CL_iter + 1)):
                    text(count_values[CL_iter])

    result = indent(
        doc.getvalue(),
        indentation=' ' * 4,
        newline='\r\n'
    )
    sync_file = open(path + "sync.xml", "w")
    sync_file.write(result)
    sync_file.close()


if __name__ == "__main__":
    # SEQ_INPUT, SEQ_DICT = seq_file_input(seq_file_name='sequences/turbo_FLASH_060924_0444.seq')
    SEQ_INPUT, SEQ_DICT = seq_file_input(seq_file_name='sequences/test1_full.seq')

    params_path = 'sequences/'
    # params_filename = "turbo_FLASH_060924_0444"
    params_filename = "test1_full"

    file = open(params_path + params_filename + ".json", 'r')
    SEQ_PARAM = json.load(file)
    file.close()
    '''
    integartion of srv_seq_gen
    '''
    # SEQ_PARAM = set_limits()
    # SEQ_INPUT = save_param()
    # SEQ_DICT = SEQ_INPUT.waveforms_export()
    '''
    simulation of inputing the JSON and SEQ
    '''

    # artificial delays due to construction of the MRI
    # искусственные задержки из-за тех. особенностей МРТ
    RF_dtime = 100 * 1e-6
    TR_dtime = 100 * 1e-6

    time_info = SEQ_INPUT.duration()
    blocks_number = time_info[1]
    time_dur = time_info[0]
    print(time_dur)
    time_step = 20 * 1e-9
    SYNC_STEP = time_step
    N_samples = int(time_dur / time_step)
    print(N_samples)
    time_sample = np.linspace(0, time_dur, N_samples)

    # output interpretation. all formats of files defined in method
    # интерпретация выхода. Все форматы файлов определены в методе
    output_seq(SEQ_DICT, SEQ_PARAM)

    # defining constants of the sequence
    # определение констант последовательности
    local_definitions = SEQ_INPUT.definitions
    ADC_raster = local_definitions['AdcRasterTime']
    RF_raster = local_definitions['RadiofrequencyRasterTime']

    gate_adc = np.zeros(N_samples)
    gate_rf = np.zeros(N_samples)
    gate_tr_switch = np.ones(N_samples)
    gate_gx = np.zeros(N_samples)
    gate_gy = np.zeros(N_samples)
    gate_gz = np.zeros(N_samples)

    local_delay_rf = RF_dtime
    local_delay_tr = TR_dtime
    local_raster_time = time_step

    RF_ASSIST = [SEQ_DICT['t_rf'][0] - RF_dtime, SEQ_DICT['t_rf'][-1]]

    synchronization(N_samples, RF_ASSIST, SEQ_DICT)
