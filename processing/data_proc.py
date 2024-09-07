import numpy as np
from matplotlib import pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000
from sklearn.metrics import mean_squared_error

data = np.genfromtxt("GrA_test1/AKIP0006DAT/AKIP0006.DAT",
                     skip_header=1,
                     skip_footer=1,
                     dtype=None,
                     delimiter=',')

initial_seq = np.genfromtxt("gradient_ampl_test_2.txt",
                     skip_header=1,
                     skip_footer=1,
                     dtype=None,
                     delimiter='')

osc_error = 0.8
time_step = 10e-6

# Задаётся шагом по амплитуде на осциллографе

def draw_osc():
    time_len = 28
    step_counts = len(data[:, 0])
    time_beg = 0
    time_end = 28
    step_beg = int(step_counts / time_len * time_beg)
    step_end = int(step_counts / time_len * time_end)
    curr = data[step_beg:step_end, 2]
    # volt = data[step_beg:step_end, 3]

    gate1 = data[step_beg:step_end, 0]
    gate2 = data[step_beg:step_end, 1]

    # plt.plot(volt, label="voltage")
    # plt.plot(curr, label="amperage")
    # # plt.plot(gate2, label="gate2")
    # # plt.plot(gate1, label="gate1")
    # plt.title('Сигнал с осциллографа')
    # plt.legend()
    # plt.savefig("proc.png")
    # plt.plot(np.transpose(initial_seq)[0]*200, np.transpose(initial_seq)[1]/32767*90)
    # plt.show()
    t_sample = np.arange(len(data[:, 0]))
    fourier_curr = np.fft.fft(curr)
    n = curr.size
    freq = np.fft.fftfreq(n, d=time_step)

    # plt.plot(freq, np.abs(fourier_curr))
    # plt.ylim([-1e4, 2e5])
    # plt.title('Абсолютное значение')
    # plt.show()
    # plt.plot(freq, np.angle(fourier_curr))
    # plt.title('Фазовая часть спектра')
    # plt.show()
    # print(freq)
    # print(fourier_curr)
    # # plt.xlim([-100, 100])
    # # plt.ylim([-9e5, 9e5])
    # plt.title('Модуль часть спектра')
    # plt.show()
    # plt.plot(freq, np.angle(fourier_curr))
    # print(len(freq))
    # print(len(fourier_curr))
    for i in range(len(freq)):
        if np.abs(freq[i]) > 250:
            fourier_curr[i] = 0
    plt.plot(freq, np.abs(fourier_curr.real))
    plt.ylim([-1e4, 2e5])
    plt.show()
    # # plt.xlim([-100, 100])
    # # plt.ylim([-9e5, 9e5])
    # plt.title('Фазовая часть спектра')
    # plt.show()
    # plt.plot(freq, fourier_curr.imag)
    # # plt.xlim([-10, 10])
    # # plt.ylim([-9e5, 9e5])
    # plt.title('Мнимая часть спектра')
    # plt.show()
    # fourier_curr[n//2-100:n//2+100] = 0
    # plt.plot(fourier_curr[n//2-10000:n//2+10000].real)
    # plt.show()
    re_inv_f = np.fft.ifft(fourier_curr).real
    plt.plot(re_inv_f)
    plt.show()

def find_begin():
    for i in range(len(data[:, 0]) - 1):
        if abs(data[i, 3] - data[i + 1, 3]) > 10:
            print("First step of voltage in the system is:", i)
            print("Current voltage:", data[i + 1, 3])
            print("Amperage for 5 steps further:", data[i + 1:i + 9, 2])
            return i


def amperage_sensor():
    amper = np.genfromtxt("Ifb_log.txt",
                          skip_header=1,
                          skip_footer=1,
                          dtype=None,
                          delimiter='\n')
    print("Number of IFB points", len(amper))
    print("Maximal amperage by:\n", "IFB:", max(amper), '\n', "Oscilloscope:",
          str(max(data[:, 2])) + "±" + str(osc_error))
    print("Minimal amperage by:\n", "IFB:", min(amper), '\n', "Oscilloscope:",
          str(round(min(data[:, 2]), 1)) + "±" + str(osc_error))
    step = 1e-5
    time_sample = []
    for i in range(len(amper)):
        time_sample.append(step*i)

    initial_seq_t = initial_seq[:, 0]*step
    initial_seq_val = initial_seq[:, 1]/32768*90

    start_point_t = 4.36
    end_point_t = 4.4

    plt.plot(initial_seq_t, initial_seq_val)
    plt.scatter(time_sample, amper, c='red', s=2)
    plt.xlim([4.3957+10, 4.3964+10])
    plt.ylim([-55,0.5])
    plt.show()

    a = np.array(amper[874000:874500])
    nulls = np.zeros_like(amper[874000:874500])
    MSE_mean = mean_squared_error(nulls, a)
    print("IFB MSE:", MSE_mean)

def draw_initial():
    plt.plot(np.transpose(initial_seq)[0], np.transpose(initial_seq)[1])
    plt.show()

# draw_osc()
# pulse_step = find_begin()
# print(len(data[:, 0]))
# amperage_sensor()
# draw_initial()


def sinc_model():
    sample = np.linspace(-4, 4, 100)
    init = np.sinc()
