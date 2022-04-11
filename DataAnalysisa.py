import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import skewnorm, norm
from scipy.special import erf


#plot data function
def plot_data(data, label=None, data2=None):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], '.', label="Experimental Data")
    if data2 is not None:
        ax.plot(data2[:, 0], data2[:, 1], '-r', label="Curve of Best Fit")
    if label:
        ax.set_title(label)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Counts")

    fig.show()

def line(x, m, b):
    return m*x + b

#args[0] is mean
#args[1] is std
#args[2] is amp
def gauss(x, *args):
    return norm.pdf(x, args[0], args[1])#args[2] * np.exp(-(x-args[0])**2 / (2 * args[1]**2))


#args[0] is mean
#args[1] is std
#args[2] is amp
#args[3] is shift
def gauss_shift(x, *args):
    return norm.pdf(x, args[0], args[1]) + args[2]


#args[0] is decay constant
#args[1] is principal
#args[2] is x shift
def exponential_decay(x, *args):
    return args[1] * np.exp(-(x/args[0]))

#f is function to fit
#data is data to fit
#num is window width
#p0 is parameter estimate
def big_fit(f, data, num=None, p0=None, label=None, plot=True):
    if num:
        max_idx = np.argmax(data[:, 1])
        if max_idx-num//2 < 0:
            print("Error")
        data_short = data[max_idx-num//2:max_idx+num//2]
    else:
        data_short = data.copy()
    #data_short_sum = data_short[:, 1].sum()
    #data_short[:, 1] = data_short[:, 1] / data_short_sum
    data_popt, data_pcov = curve_fit(f, data_short[:, 0], data_short[:, 1], p0=p0, maxfev=10000)
    data_perr = np.sqrt(np.diag(data_pcov))
    print("{} parameter values".format(label))
    for idx in range(len(data_popt)):
        print("arg[{}]: {:.5f} +/- {:.5f}".format(idx, data_popt[idx], data_perr[idx]))
    print()
    if plot:
        span_size = len(data_short) * 10
        data_fit = np.zeros((span_size, 2))
        data_fit[:, 0] = np.linspace(data_short[:, 0].min(), data_short[:, 0].max(), num=span_size)
        data_fit[:, 1] = f(data_fit[:, 0], *data_popt.tolist())
        plot_data(data_short, data2=data_fit, label=label)
    return data_popt

def big_fit_am(f, data, num=None, p0=None, label=None, plot=True):
    if num:
        max_idx = np.argmax(data[:, 1])
        if max_idx-num//2 < 0:
            print("Error")
        data_short = data[max_idx-num//2:max_idx+num//2]
    else:
        data_short = data.copy()
    #data_short_sum = data_short[:, 1].sum()
    #data_short[:, 1] = data_short[:, 1] / data_short_sum
    sigma = np.ones(len(data_short[:, 1])) * 5
    data_popt, data_pcov = curve_fit(f, data_short[:, 0], data_short[:, 1], p0=p0, sigma=sigma, maxfev=10000)
    data_perr = np.sqrt(np.diag(data_pcov))
    print("{} parameter values")
    for idx in range(len(data_popt)):
        print("arg[{}]: {:.5f} +/- {:.5f}".format(idx, data_popt[idx], data_perr[idx]))
    print()

    if plot:
        span_size = len(data_short) * 10
        data_fit = np.zeros((span_size, 2))
        data_fit[:, 0] = np.linspace(data_short[:, 0].min(), data_short[:, 0].max(), num=span_size)
        data_fit[:, 1] = f(data_fit[:, 0], *data_popt.tolist())
        #data_fit[:, 1] = f(data_fit[:, 0], 68.0/1000, 60)
        plot_data(data_short, data2=data_fit, label=label)
    return data_popt

def calibrate_time(time_calibration, time_span):
    time_max = np.array([cell for cell in time_calibration if cell[1] > 150])
    times = np.array([0.08, 0.16, 0.32])
    time_max[:, 1] = times

    #Convert voltage to time delay
    time_fit_popt, time_fit_pcov = curve_fit(line, time_max[:, 0], time_max[:, 1])
    times = line(time_span, time_fit_popt[0], time_fit_popt[1])
    times = times * 1000.0
    return times

#fit cable time delay
def fit_cable(cable_data, num=None, p0=[1, 1]):
    popt = big_fit(gauss, cable_data, num=num, p0=p0, label="Cable time delay")

#curve fit am-241
def fit_am(am_data, start=None, stop=None, num=30, p0=[0.07,140], cable_delay=None):
    data = am_data.copy()
    start_idx = np.searchsorted(data[:, 0], start, side='left')
    end_idx = np.searchsorted(data[:, 0], stop, side='left')
    data = data[start_idx:end_idx]
    plot_data(data, label="Short am-241")
    data[:, 0] = data[:, 0] - cable_delay
    popt = big_fit_am(exponential_decay, data, label="Am-241", p0=p0)

#curve fit am-241
def fit_am_linear(am_data, start=None, stop=None, num=30, p0=[0.07,140], cable_delay=None):
    data = am_data.copy()
    start_idx = np.searchsorted(data[:, 0], start, side='left')
    end_idx = np.searchsorted(data[:, 0], stop, side='left')
    data = data[start_idx:end_idx]
    plot_data(data, label="Short am-241")
    max = data[0, 1]
    data[:, 0] = data[:, 0] - cable_delay
    data[:, 1] = data[:, 1] / max
    data[:, 1] = np.log(data[:, 1])
    popt = big_fit_am(line, data, label="Am-241", p0=p0)

#curve fit na-241
def fit_na(na_data, num=50, p0=[1,1]):
    popt = big_fit(gauss, na_data, label="Na-22", p0=p0)

def test_gauss():
    num_points = 100
    data_test = np.zeros((num_points, 2))
    data_test[:, 0] = np.random.rand(num_points) * 10
    data_test[:, 1] = gauss(data_test[:, 0], 5, 1) + np.random.rand(num_points) * 0.01

    p0_gauss = [1, 1]
    popt = big_fit(gauss, data_test, label="Gauss Test", p0=p0_gauss)

def test_decay(crop=False):
    num_points = 100
    data_test = np.zeros((num_points, 2))
    data_test[:, 0] = np.linspace(0, 10, num=num_points)
    data_test[:, 1] = exponential_decay(data_test[:, 0], 5, 250) + np.random.rand(num_points) * 0.01

    if crop:
        start_idx = np.searchsorted(data_test[:, 0], 2, side='left')
        end_idx = np.searchsorted(data_test[:, 0], 9, side='left')
        data_test = data_test[start_idx:end_idx]
    p0_decay = [6, 260]
    popt = big_fit(exponential_decay, data_test, label="Exponential Decay Test", p0=p0_decay)


def run_data_old():
    time_calibration = np.genfromtxt("Data/good_calibration_0.16_0.32_0.64.csv", delimiter=',')
    am_241 = np.genfromtxt("Data/am-241.csv", delimiter=',')
    cable_old = np.genfromtxt("Data/long_cable time difference.csv", delimiter=',')
    na_22 = np.genfromtxt("Data/Na-22_332.csv", delimiter=',')

    plot_data(time_calibration, 'Time Calibration')
    plot_data(am_241, 'Am-241')
    plot_data(cable_old, 'Long Cable')
    plot_data(na_22, "Na-22")

    times = calibrate_time(time_calibration, am_241[:, 0])
    cable_old[:, 0] = times
    na_22[:, 0] = times

    fit_cable(cable_old)
    fit_am(am_241, start=0.48, stop=0.6)
    fit_na(na_22)

def run_data_new():
    time_calibration = np.genfromtxt("Data/time_calibration_new.csv", delimiter=',')
    am_241 = np.genfromtxt("Data/am_241_83127_mod.csv", delimiter=',')
    cable_new = np.genfromtxt("Data/long_cable_good.csv", delimiter=',')

    plot_data(time_calibration, 'Time Calibration')
    plot_data(am_241[233:250], 'Am-241')
    plot_data(cable_new, 'Long Cable')


    times = calibrate_time(time_calibration, am_241[:, 0])
    cable_new[:, 0] = times
    am_241[:, 0] = times

    cable_peak_idx = np.argmax(cable_new[:, 1])
    cable_delay = cable_new[cable_peak_idx, 0]

    plot_data(am_241, "Am-241 calibrated")
    peak = np.argmax(am_241[:, 1])
    peak_time = am_241[peak, 0]



    fit_cable(cable_new)
    fit_am_linear(am_241, start=226, stop=236, cable_delay=cable_delay)
    fit_am(am_241, start=226, stop=236, cable_delay=cable_delay)



def test_all():
    test_gauss()
    test_decay()
    test_decay(crop=True)

if __name__ == "__main__":
    run_data_new()










