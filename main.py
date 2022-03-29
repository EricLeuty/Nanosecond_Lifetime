import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

time_calibration = np.genfromtxt("Data/good_calibration_0.16_0.32_0.64.csv", delimiter=',')
am_241 = np.genfromtxt("Data/am-241.csv", delimiter=',')
cable = np.genfromtxt("Data/long_cable time difference.csv", delimiter=',')
na_22 = np.genfromtxt("Data/Na-22_332.csv", delimiter=',')
time_am = 164719
time_cable = 79

def plot_data(data, label=None, data2=None):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], '.')
    if data2:
        ax.plot(data[:, 0], data2, '-r')
    if label:
        ax.set_title(label)
    fig.show()

plot_data(time_calibration, 'Time Calibration')
plot_data(am_241, 'Am-241')
plot_data(cable, 'Long Cable')
plot_data(na_22, "Na-22")

time_max = np.array([cell for cell in time_calibration if cell[1] > 100])
times = np.array([0.16, 0.32, 0.64])
time_max[:, 1] = times

def line(x, m, b):
    return m*x + b

def gauss(x, mean, std, amp, shift):
    return amp * np.exp(-(x-mean)**2 / (2 * std**2)) + shift


time_fit_popt, time_fit_pcov = curve_fit(line, time_max[:, 0], time_max[:, 1])
times = line(am_241[:, 0], time_fit_popt[0], time_fit_popt[1])
am_241[:, 0] = times
plot_data(am_241, 'Am-241 wrt time')

maximum = np.flip(np.argsort(am_241[:, 1]))

am_241_short = am_241[maximum[:30]]

plot_data(am_241_short, 'Shortened Am-241')

am_241_popt, am_241_pcov = curve_fit(gauss, am_241_short[:, 0], am_241_short[:, 1], p0=[0.15, 100, 850, 290])
print(am_241_popt)
time_range = np.linspace(am_241_short[:,0].min(), am_241_short[:,0].max())
am_241_fit = gauss(time_range, am_241_popt[0], am_241_popt[1], am_241_popt[2], am_241_popt[3])

fitted_data = np.zeros((50, 2))
fitted_data[:, 0] = time_range
fitted_data[:, 1] = am_241_fit
plot_data(fitted_data, "Am-241 Gaussian")






