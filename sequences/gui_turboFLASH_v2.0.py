# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:16:14 2024

@author: zilya
"""

import tkinter as tk
import numpy as np
import json
from datetime import datetime
from seqgen_turboFLASH import seqgen_turboFLASH
from types import SimpleNamespace

import os
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)

from utilities.param_digits import param_digits_create
from utilities.param_digits import param_dj_get
from utilities.param_digits import param_set_lim
from utilities.param_digits import read_json
from utilities.param_moduls import add_module

from utilities.param_constants import param_hardware
from utilities.param_constants import param_rf_GRE
from utilities.param_constants import param_rf_inv
from utilities.param_constants import param_default


def calc_lims():
    # reading params from GUI
    global param
    param = param_dj_get(M0_dj, param_list)

    # reading hardware constants
    hw = param_hardware()
    rf = param_rf_GRE()
    rf = SimpleNamespace(**rf.__dict__)
    # calculating limits
    lim = SimpleNamespace()

    Np_ph_over = param.Np_image * (1 + param.ph_over_phase / 100)
    param.Np = round(Np_ph_over)
    param.ph_over = (Np_ph_over / param.Np_image - 1) * 100
    param.FoV_p = param.FoV_p_image * (1 + param.ph_over_phase / 100)

    rf.BW_ex_pulse = rf.t_BW_product_ex / rf.t_ex  # Hz. BW импульса

    param.TE = np.ceil(param.TE / hw.grad_raster_time) * hw.grad_raster_time
    param.TR = np.ceil(param.TR / hw.grad_raster_time) * hw.grad_raster_time

    t_read = 1 / param.BW_pixel  # Время сбора эхо сигнала
    #tau_ex = BW_ex_pulse/(sl_thkn*G_slew_max) # Время нарастания сс градиента 
    #tau_r = Nf/(FoV_f*t_read*G_slew_max)      # Время нарастания считывающего градиента 
    t_sps = param.spoil_strenght / (
                param.sl_thkn * hw.G_amp_max) + hw.tau_max  # Длительность сс спойлерного градиента, который в конце
    t_spf = param.spoil_strenght * param.Nf / (
                param.FoV_f * hw.G_amp_max) + hw.tau_max  # Длительность спойлерного градиента по направлению фазового код-ия

    A_ex = rf.BW_ex_pulse * (rf.t_ex + hw.tau_max) / param.sl_thkn
    A_read = param.Nf * (t_read + hw.tau_max) / param.FoV_f / t_read
    A_ref = 0.5 * A_ex
    A_pre = 0.5 * A_read
    A_ph_max = param.Np / (2 * param.FoV_p)

    max_blip_area = hw.G_amp_max * hw.tau_max
    if A_ref <= max_blip_area:
        t_ref = 2.0 * hw.tau_max  #In s
    else:
        t_ref = 2.0 * hw.tau_max + (A_ref - max_blip_area) / hw.G_amp_max  #In s

    if A_pre <= max_blip_area:
        t_pre = 2.0 * hw.tau_max  #In s
    else:
        t_pre = 2.0 * hw.tau_max + (A_pre - max_blip_area) / hw.G_amp_max  #In s

    if A_ph_max <= max_blip_area:
        t_ph = 2.0 * hw.tau_max  #In s
    else:
        t_ph = 2.0 * hw.tau_max + (A_ph_max - max_blip_area) / hw.G_amp_max  #In s

    # создание delay в начале каждой последовательности
    if hw.tau_max > hw.rf_dead_time:
        delay = 0
    else:
        delay = hw.rf_dead_time - hw.tau_max
        delay = np.ceil(delay / hw.grad_raster_time) * hw.grad_raster_time

    # Подгон под растр  
    t_read = np.ceil(t_read / hw.grad_raster_time) * hw.grad_raster_time
    t_ref = np.ceil(t_ref / hw.grad_raster_time) * hw.grad_raster_time
    t_sps = np.ceil(t_sps / hw.grad_raster_time) * hw.grad_raster_time
    t_spf = np.ceil(t_spf / hw.grad_raster_time) * hw.grad_raster_time
    t_ref = np.ceil(t_ref / hw.grad_raster_time) * hw.grad_raster_time
    t_pre = np.ceil(t_pre / hw.grad_raster_time) * hw.grad_raster_time
    t_ph = np.ceil(t_ph / hw.grad_raster_time) * hw.grad_raster_time
    param.BW_pixel = 1 / t_read

    # Максимальная длительность среди градиентов после импульса
    t_ref_block = max(t_ref, t_pre, t_ph)
    if t_ref_block == 2 * hw.tau_max:
        t_ref_block = t_ref_block + hw.grad_raster_time
    # Максимальная длительность среди последних градиентов, начинающихся сразу после считывания
    tsp = max(t_sps, t_spf)

    t_ref_block_max = param.TE - (hw.tau_max + rf.t_ex / 2 + t_read / 2 + hw.adc_dead_time)
    tsp_max = param.TR - (2.0 * hw.tau_max + rf.t_ex / 2 + param.TE + t_read / 2)
    t_ref_block_max = np.floor(t_ref_block_max / hw.grad_raster_time) * hw.grad_raster_time
    tsp_max = np.floor(tsp_max / hw.grad_raster_time) * hw.grad_raster_time
    # Вычисление лимитов
    lim.TE_min = rf.t_ex / 2 + 2 * hw.tau_max + t_ref_block + t_read / 2
    TE_max1 = param.TR - (rf.t_ex / 2 + t_read / 2 + 2 * hw.tau_max + tsp)
    TE_max2 = 1
    lim.TE_max = min(TE_max1, TE_max2)
    lim.TR_min = param.TE + rf.t_ex / 2 + t_read / 2 + 2 * hw.tau_max + tsp + delay
    lim.TR_max = 10

    lim.TE_min = np.ceil(lim.TE_min / hw.grad_raster_time) * hw.grad_raster_time
    lim.TE_max = np.floor(lim.TE_max / hw.grad_raster_time) * hw.grad_raster_time
    lim.TR_min = np.ceil(lim.TR_min / hw.grad_raster_time) * hw.grad_raster_time
    lim.TR_max = np.floor(lim.TR_max / hw.grad_raster_time) * hw.grad_raster_time

    lim.Nf_min = 1
    Nf_max1 = param.FoV_f * hw.G_amp_max * t_read
    Nf_max2 = param.FoV_f * hw.G_amp_max * (t_ref_block_max - hw.tau_max) * t_read / (0.5 * (t_read + hw.tau_max))
    Nf_max3 = param.FoV_f * hw.G_amp_max * (tsp_max - hw.tau_max) / (param.spoil_strenght)
    lim.Nf_max = min(Nf_max1, Nf_max2, Nf_max3)
    lim.Np_min = 1
    lim.Np_max = lim.Nf_max

    FoV_min = 25e-3
    FoV_f_min1 = param.Nf / (hw.G_amp_max * t_read)
    FoV_f_min2 = param.Nf * 0.5 * (t_read + hw.tau_max) / (hw.G_amp_max * (t_ref_block_max - hw.tau_max) * t_read)
    FoV_f_min3 = param.Nf * param.spoil_strenght / (hw.G_amp_max * (tsp_max - hw.tau_max))
    lim.FoV_f_min = max(FoV_f_min1, FoV_f_min2, FoV_f_min3, FoV_min)
    lim.FoV_f_max = 450e-3

    lim.FoV_p_min = FoV_min
    lim.FoV_p_max = lim.FoV_f_max

    lim.Np_image_min = lim.Np_min
    lim.Np_image_max = lim.Np_max / (1 + param.ph_over_phase / 100)

    lim.FoV_p_image_min = lim.FoV_p_min / (1 + param.ph_over_phase / 100)
    lim.FoV_p_image_max = lim.FoV_p_max / (1 + param.ph_over_phase / 100)

    lim.BW_pixel_min = 1 / (2 * (param.TE - rf.t_ex / 2 - t_ref_block - 2 * hw.tau_max))
    BW_pixel_max1 = param.FoV_f * hw.G_amp_max / param.Nf
    BW_pixel_max2 = 1780
    lim.BW_pixel_max = min(BW_pixel_max1, BW_pixel_max2)

    lim.sl_thkn_min = rf.BW_ex_pulse / hw.G_amp_max
    lim.sl_thkn_max = 20e-3

    lim.sl_nb_min, lim.sl_nb_max = 1, 110
    lim.sl_gap_min, lim.sl_gap_max = 1, 110

    lim.FA_min, lim.FA_max = 1, 90
    lim.ph_over_phase_min, lim.ph_over_phase_max = 0, 100
    lim.spoil_strenght_min, lim.spoil_strenght_max = 1, 10
    lim.RF_spoil_min, lim.RF_spoil_max = 1, 180
    lim.average_min, lim.average_max = 1, 10
    lim.D_scans_min, lim.D_scans_max = 0, 100

    param.IR = True
    param.magn_prep = "IR"
    for_module = SimpleNamespace()
    for_module.IR_sel = False
    for_module.TR_min_true = lim.TR_min
    for_module.A_ex = A_ex
    for_module.t_ex = rf.t_ex
    for_module.N_TR = param.Np
    param.dG = hw.tau_max

    param, rf, lim = add_module(param, rf, hw, for_module, lim)

    # showing limits on GUI
    param_set_lim(M0_dj, param_list, lim, param)

    # database of params
    param = SimpleNamespace(**param.__dict__, **hw.__dict__, **rf.__dict__)


def save():
    output_filename = str(M0_dj.textBox_name.get())
    file = open(output_filename + '.json', 'w')
    json.dump(param.__dict__, file, indent=4)
    file.close()

    output_sequence = seqgen_turboFLASH(param)
    output_sequence.plot(save=True)
    output_sequence.write(output_filename)


def read_js():
    read_json(Main, M0_dj, param_list)


name = "turbo_FLASH"

default = param_default()
default.sl_nb = 1
default.sl_thkn = 5e-3
default.sl_gap = 100

default.FoV_f = 64e-3
default.FoV_p_image = 64e-3
default.Nf = 32
default.Np_image = 32
default.BW_pixel = 500

default.TE = 4e-3
default.TR = 10e-3

default.FA = 90
default.RF_spoil = 117

default.average = 1
default.ph_over_phase = 0

# param_list = ['sl_nb', 'sl_thkn', 'sl_gap', 'slab_thkn',
#               'FoV_f', 'FoV_p_image', 'Nf', 'Np_image', 'Nss', 'BW_pixel',
#               'TE', 'N_TE', 'IE', 'ETL', 'contrasts','concats', 'TR', 'TD', 
#               'alpha', 'beta', 'spoil_strenght', 'RF_spoil',
#               'average', 'D_scans', 'ph_over_phase', 'ph_over_slice', 'part_fourier_factor', 'ZF',
#               'TI', 'N_TI', 'b', 'flow_comp', 'sat_sl_thkn', 'distance_ex_sat',
#               'FS', 'SPAIR', 'WS', 'WE',
#              'IR', 'DIR', 'TIR', 'SR', 'T2_prep_IR']

param_list = ['sl_nb', 'sl_thkn', 'sl_gap',
              'FoV_f', 'FoV_p_image', 'Nf', 'Np_image', 'BW_pixel',
              'TE', 'TR',
              'FA', 'RF_spoil', 'spoil_strenght',
              'average', 'ph_over_phase', 'part_fourier_factor_phase', 'ZF', 'D_scans',
              'TI'
              # 'TEeff',
              # 'FS','SPAIR','WE',
              # 'IR','DIR','t2prepIR_sel','t2prepIR_nonsel'
              ]

Main = tk.Tk()
Main.title(name)
Main.geometry("900x750+100+30")
Main.resizable(False, False)

M0_dj = SimpleNamespace()

param_digits_create(Main, M0_dj, param_list, default)

M0_dj.btn_set = tk.Button(Main, text='Set', command=calc_lims)
M0_dj.btn_set.grid(row=1, column=8, sticky="W")
M0_dj.btn_save = tk.Button(Main, text='Read json', command=read_js)
M0_dj.btn_save.grid(row=2, column=8, sticky="W")

M0_dj.btn_save = tk.Button(Main, text='Save', command=save)
M0_dj.btn_save.grid(row=4, column=8, sticky="W")
# filename
M0_dj.label_name = tk.Label(Main, text='Output file name', width=13)
M0_dj.label_name.grid(row=5, column=8, sticky="W")
M0_dj.textBox_name = tk.Entry(Main, width=25)
M0_dj.textBox_name.insert(0, name + datetime.now().strftime("_%d%m%y_%H%M"))
M0_dj.textBox_name.grid(row=6, column=8, columnspan=3, sticky="W")

calc_lims()

Main.mainloop()

# Main.mainloop()
