# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:53:21 2023

@author: zilya
"""

import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from testing.sequences.seqgen.seqgen_GRAD_TEST import *
from datetime import datetime
import json


def set_limits():
    # Задание общих аппаратных характкристик
    gamma = 42.576e6  # Hz/T    Гиромагнитное отношение водорода
    G_amp_max_mT_m = 10  # mT/m.   Максимальный градиент
    G_amp_max = G_amp_max_mT_m * 1e-3 * gamma  # Hz/m.   Максимальный градиент
    G_slew_max_T_m_s = 30  # T/m/s.  Максимальная скорость нарастания
    G_slew_max = G_slew_max_T_m_s * gamma  # Hz/m/s. Максимальная скорость нарастания
    rf_raster_time = 1e-6  # s.      Растр РЧ импульса
    grad_raster_time = 10e-6  # s.      Растр градиентов
    tau_max = G_amp_max / G_slew_max  # s.      Максимальное время нарастания градиента с учетом макс скорости нарастания
    tau_max = np.ceil(tau_max / grad_raster_time) * grad_raster_time

    # Чтение заданных в интерфэйс значений 
    AU = int(M0_dj.textBox1.get())
    NA = int(M0_dj.textBox2.get())
    t_grad = float(M0_dj.textBox3.get())
    delay_1 = float(M0_dj.textBox4.get())
    delay_2 = float(M0_dj.textBox5.get())

    if M0_dj.radio6.get() == 1:
        grad1_pol = 1
    if M0_dj.radio6.get() == 2:
        grad1_pol = -1
    if M0_dj.radio7.get() == 1:
        grad2_pol = 1
    if M0_dj.radio7.get() == 2:
        grad2_pol = -1

    N_grad = 128
    grad_step = G_amp_max / N_grad
    grad_amp = grad_step * AU

    AU_min, AU_max = 0, 128
    NA_min, NA_max = 1, 128
    t_grad_min, t_grad_max = 1e-3, 50e-3
    delay_1_min, delay_1_max = 1e-3, 50e-3
    delay_2_min, delay_2_max = 1e-3, 50e-3

    M0_dj.label1_1.configure(text=str(AU_min))
    M0_dj.label1_2.configure(text=str(AU_max))
    M0_dj.label1_3.configure(text=str(round(grad_amp / (1e-3 * gamma), 3)))
    M0_dj.label2_1.configure(text=str(NA_min))
    M0_dj.label2_2.configure(text=str(NA_max))
    M0_dj.label3_1.configure(text=str(t_grad_min))
    M0_dj.label3_2.configure(text=str(t_grad_max))
    M0_dj.label4_1.configure(text=str(delay_1_min))
    M0_dj.label4_2.configure(text=str(delay_1_max))
    M0_dj.label5_1.configure(text=str(delay_2_min))
    M0_dj.label5_2.configure(text=str(delay_2_max))

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
    param.delay_1 = delay_1
    param.delay_2 = delay_2
    param.grad1_pol = grad1_pol
    param.grad2_pol = grad2_pol


def save_param():
    output_filename = str(M0_dj.textBox17.get())
    directory_name = "sequences/"
    output_sequence = seqgen_GRAD_TEST(param, output_filename)
    # output_sequence.plot()
    output_sequence.write(directory_name+output_filename)
    file = open(directory_name+output_filename + ".json", 'w')
    json.dump(param.__dict__, file, indent=4)
    file.close()


### Default values ###

AU = 128
NA = 128
t_grad = 10e-3
delay_1 = 20e-3
delay_2 = 20e-3

win = tk.Tk()
win.title('GRAD_TEST')
win.geometry("520x420+100+100")
win.resizable(False, False)

# создаем набор вкладок
notebook = ttk.Notebook()
notebook.pack(expand=True, fill='both')

Main = tk.Frame(notebook)
Main.pack(fill='both', expand=True)
notebook.add(Main, text="Main")

M0_dj = SimpleNamespace()

M0_dj.label0_1 = tk.Label(Main, text="min", width=10)
M0_dj.label0_1.grid(row=0, column=2)
M0_dj.label0_2 = tk.Label(Main, text="max", width=10)
M0_dj.label0_2.grid(row=0, column=3)

tk.Label(Main, text='A.U.').grid(row=1, column=0, sticky="E")
M0_dj.textBox1 = tk.Entry(Main, width=7)
M0_dj.textBox1.insert(0, AU)
M0_dj.textBox1.grid(row=1, column=1)
M0_dj.label1_1 = tk.Label(Main)
M0_dj.label1_1.grid(row=1, column=2)
M0_dj.label1_2 = tk.Label(Main)
M0_dj.label1_2.grid(row=1, column=3)

tk.Label(Main, text='Grad amp (Hz/m) = ').grid(row=2, column=0, sticky="E")
M0_dj.label1_3 = tk.Label(Main)
M0_dj.label1_3.grid(row=2, column=1)

tk.Label(Main, text='NA').grid(row=3, column=0, sticky="E")
M0_dj.textBox2 = tk.Entry(Main, width=7)
M0_dj.textBox2.insert(0, NA)
M0_dj.textBox2.grid(row=3, column=1)
M0_dj.label2_1 = tk.Label(Main)
M0_dj.label2_1.grid(row=3, column=2)
M0_dj.label2_2 = tk.Label(Main)
M0_dj.label2_2.grid(row=3, column=3)

tk.Label(Main, text='Grad duration, c').grid(row=4, column=0, sticky="E")
M0_dj.textBox3 = tk.Entry(Main, width=7)
M0_dj.textBox3.insert(0, t_grad)
M0_dj.textBox3.grid(row=4, column=1)
M0_dj.label3_1 = tk.Label(Main)
M0_dj.label3_1.grid(row=4, column=2)
M0_dj.label3_2 = tk.Label(Main)
M0_dj.label3_2.grid(row=4, column=3)

tk.Label(Main, text='Delay 1, c').grid(row=5, column=0, sticky="E")
M0_dj.textBox4 = tk.Entry(Main, width=7)
M0_dj.textBox4.insert(0, delay_1)
M0_dj.textBox4.grid(row=5, column=1)
M0_dj.label4_1 = tk.Label(Main)
M0_dj.label4_1.grid(row=5, column=2)
M0_dj.label4_2 = tk.Label(Main)
M0_dj.label4_2.grid(row=5, column=3)

tk.Label(Main, text='Delay 2, c').grid(row=6, column=0, sticky="E")
M0_dj.textBox5 = tk.Entry(Main, width=7)
M0_dj.textBox5.insert(0, delay_2)
M0_dj.textBox5.grid(row=6, column=1)
M0_dj.label5_1 = tk.Label(Main)
M0_dj.label5_1.grid(row=6, column=2)
M0_dj.label5_2 = tk.Label(Main)
M0_dj.label5_2.grid(row=6, column=3)

M0_dj.radio6 = tk.IntVar()
M0_dj.radio6.set(1)
M0_dj.label6 = tk.Label(Main, text="Grad1 polarity:", width=10)
M0_dj.label6.grid(row=7, column=0, sticky="E")
M0_dj.R6_1 = tk.Radiobutton(Main, text="+", variable=M0_dj.radio6, value=1)
M0_dj.R6_1.grid(row=7, column=1)
M0_dj.R6_2 = tk.Radiobutton(Main, text="-", variable=M0_dj.radio6, value=2)
M0_dj.R6_2.grid(row=7, column=2)

M0_dj.radio7 = tk.IntVar()
M0_dj.radio7.set(1)
M0_dj.label7 = tk.Label(Main, text="Grad2 polarity:", width=10)
M0_dj.label7.grid(row=8, column=0, sticky="E")
M0_dj.R7_1 = tk.Radiobutton(Main, text="+", variable=M0_dj.radio7, value=1)
M0_dj.R7_1.grid(row=8, column=1)
M0_dj.R7_2 = tk.Radiobutton(Main, text="-", variable=M0_dj.radio7, value=2)
M0_dj.R7_2.grid(row=8, column=2)

set_limits()

M0_dj.btn1 = tk.Button(Main, text='Set', command=set_limits, width=10)
M0_dj.btn1.grid(row=1, column=5, sticky="W")

M0_dj.btn1 = tk.Button(Main, text='Save', command=save_param, width=10)
M0_dj.btn1.grid(row=4, column=5, sticky="W")

# filename
M0_dj.label17 = tk.Label(Main, text='Output file name', width=13)
M0_dj.label17.grid(row=6, column=5, sticky="W")
M0_dj.textBox17 = tk.Entry(Main, width=25)
M0_dj.textBox17.insert(0, "GRAD_TEST_" + datetime.now().strftime("%d%m%y_%H%M"))
M0_dj.textBox17.grid(row=7, column=5, columnspan=3, sticky="W")

win.mainloop()
