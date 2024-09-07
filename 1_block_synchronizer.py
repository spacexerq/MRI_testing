# synchronizer : converts Pulseq (.seq) files into sequences of amplitude, time and synchro sets.
# Output is given by MR scanner
# Babich Nikita, Kozin Roman, Karsakov  Grigory
# March 2024

import numpy as np
from matplotlib import pyplot as plt
# from pulseq_fixed import sequence_fixed as puls_fix
import pypulseq as puls_fix


def output_seq(dict):
    """
    The interpretation from pypulseq format of sequence to the files needed to analog part of MRI
    Интерпретация последовательности из формата pypulseq в файлы, необходимые для аналоговой части МРТ.
    :param dict: Dictionary of the impulse sequence pypulseq provided
        Словарь импульсных последовательностей, предоставленный pypulseq
    :return: files in "data_output_seq/" directory of every type of amplitudes and time points
        Запись в файлы в директории "data_output_seq/" амплитуд и времени для всех типов
    """
    loc_t_adc = dict['t_adc']
    loc_t_rf = dict['t_rf']
    loc_t_rf_centers = dict['t_rf_centers']
    loc_t_gx = dict['t_gx']
    loc_t_gy = dict['t_gy']
    loc_t_gz = dict['t_gz']
    loc_adc = dict['adc']
    loc_rf = dict['rf']
    loc_rf_centers = dict['rf_centers']
    loc_gx = dict['gx']
    loc_gy = dict['gy']
    loc_gz = dict['gz']
    with open('data_output_seq/t_adc.txt', 'w') as f:
        data = str(tuple(loc_t_adc))
        f.write(data)
    with open('data_output_seq/t_rf.txt', 'w') as f:
        data = str(tuple(loc_t_rf))
        f.write(data)
    with open('data_output_seq/t_rf_centers.txt', 'w') as f:
        data = str(tuple(loc_t_rf_centers))
        f.write(data)
    with open('data_output_seq/t_gx.txt', 'w') as f:
        data = str(tuple(loc_t_gx))
        f.write(data)
    with open('data_output_seq/t_gy.txt', 'w') as f:
        data = str(tuple(loc_t_gy))
        f.write(data)
    with open('data_output_seq/t_gz.txt', 'w') as f:
        data = str(tuple(loc_t_gz))
        f.write(data)
    with open('data_output_seq/adc.txt', 'w') as f:
        data = str(tuple(loc_adc))
        f.write(data)
    with open('data_output_seq/rf.txt', 'w') as f:
        data = str(tuple(loc_rf))
        f.write(data)
    with open('data_output_seq/rf_centers.txt', 'w') as f:
        data = str(tuple(loc_rf_centers))
        f.write(data)
    with open('data_output_seq/gx.txt', 'w') as f:
        data = str(tuple(loc_gx))
        f.write(data)
    with open('data_output_seq/gy.txt', 'w') as f:
        data = str(tuple(loc_gy))
        f.write(data)
    with open('data_output_seq/gz.txt', 'w') as f:
        data = str(tuple(loc_gz))
        f.write(data)


def adc_correction():
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


# gates_output(gates_release)


if __name__ == '__main__':
    print('')
    seq_file = "seq_store/SE_rfdeath_5000.seq"
    seq_input = puls_fix.Sequence()
    seq_input.read(file_path=seq_file)
    seq_output_dict = seq_input.waveforms_export(time_range=(0, 3))

    # artificial delays due to construction of the MRI
    # искусственные задержки из-за тех. особенностей МРТ
    RF_dtime = 100 * 1e-6
    TR_dtime = 100 * 1e-6

    time_info = seq_input.duration()
    blocks_number = time_info[1]
    time_dur = time_info[0]
    time_step = 20 * 1e-9
    N_samples = int(time_dur / time_step)
    time_sample = np.linspace(0, time_dur, N_samples)

    # output interpretation. all formats of files defined in method
    # интерпретация выхода. Все форматы файлов определены в методе
    output_seq(seq_output_dict)

    # defining constants of the sequence
    # определение констант последовательности
    local_definitions = seq_input.definitions
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

    RF_assintant = [seq_output_dict['t_rf'][0] - RF_dtime, seq_output_dict['t_rf'][-1]]

    synchronization(N_samples)

    # testing plots for synchronization
    # графики тестов синхрозации
    plt.plot(seq_output_dict['t_gx'][:int(N_samples)], seq_output_dict['gx'][:int(N_samples)])
    plt.plot(seq_output_dict['t_gy'][:int(N_samples)], seq_output_dict['gy'][:int(N_samples)])
    plt.plot(seq_output_dict['t_gz'][:int(N_samples)], seq_output_dict['gz'][:int(N_samples)])
    plt.savefig("plots_output/gradients.png")
    plt.show()

    plt.plot(seq_output_dict['t_gx'][:int(N_samples)], seq_output_dict['gx'][:int(N_samples)] / 720)
    plt.plot(time_sample[:int(N_samples)], gate_adc[:int(N_samples)], label='ADC gate')
    plt.plot(time_sample[:int(N_samples)], gate_tr_switch[:int(N_samples)], label='TR switch')
    plt.plot(seq_output_dict['t_rf'], seq_output_dict['rf'] / 210, label='RF signal')
    plt.plot(time_sample[:int(N_samples)], gate_rf[:int(N_samples)], label='RF gate')
    plt.legend()
    plt.savefig("plots_output/synchro_pulse.png")
    plt.show()
