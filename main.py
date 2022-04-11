import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import skewnorm, norm
from scipy.special import erf

RUN = True

time_calibration = np.genfromtxt("Data/good_calibration_0.16_0.32_0.64.csv", delimiter=',')
am_241 = np.genfromtxt("Data/am-241.csv", delimiter=',')
cable = np.genfromtxt("Data/long_cable time difference.csv", delimiter=',')
na_22 = np.genfromtxt("Data/Na-22_332.csv", delimiter=',')
time_am = 164719
time_cable = 79

#plot data function
def plot_data(data, label=None, data2=None):
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], data[:, 1], '.')
    if data2 is not None:
        ax.plot(data2[:, 0], data2[:, 1], '-r')
    if label:
        ax.set_title(label)
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

def skew(x, *args):
    return skewnorm.pdf(x, args[2], args[0], args[1])

def exponential_decay(x, *args):
    return args[2] * np.exp(-((x - args[1])/args[0]))

def big_fit(f, data, num=None, p0=None, label=None):
    if num:
        max_idx = np.argmax(data[:, 1])
        if max_idx-num//2 < 0:
            print("Error")
        data_short = data[max_idx-num//2:max_idx+num//2]
    else:
        data_short = data
    data_popt, data_pcov = curve_fit(f, data_short[:, 0], data_short[:, 1], p0=p0, maxfev=10000)

    data_short_sum = data_short[:, 1].sum()
    data_short[:, 1] = data_short[:, 1] / data_short_sum

    print("{} time: {:.5f} +/- {:.5f}".format(label, data_popt[0], data_popt[1]))
    span_size = len(data_short) * 10
    data_fit = np.zeros((span_size, 2))
    data_fit[:, 0] = np.linspace(data_short[:, 0].min(), data_short[:, 0].max(), num=span_size)
    data_fit[:, 1] = f(data_fit[:, 0], *data_popt.tolist()) / data_short_sum
    plot_data(data_short, data2=data_fit, label=label)

plot_data(time_calibration, 'Time Calibration')
plot_data(am_241, 'Am-241')
plot_data(cable, 'Long Cable')
plot_data(na_22, "Na-22")


time_max = np.array([cell for cell in time_calibration if cell[1] > 100])
times = np.array([0.16, 0.32, 0.64])
time_max[:, 1] = times



#Convert voltage to time delay
time_fit_popt, time_fit_pcov = curve_fit(line, time_max[:, 0], time_max[:, 1])
times = line(am_241[:, 0], time_fit_popt[0], time_fit_popt[1])
am_241[:, 0] = times
cable[:, 0] = times
na_22[:, 0] = times

#fit cable time delay
num_cable = 20
p0_cable = [1, 1]
if RUN:
    big_fit(gauss, cable, num=num_cable, p0=p0_cable, label="Cable time delay")

#curve fit am-241
num_am = 30
p0_am = [1, 1]
if RUN:
    big_fit(gauss, am_241, num=num_am, label="Am-241", p0=p0_am)

#curve fit na-241
num_na = 50
p0_na = [1, 1]
if RUN:
    big_fit(gauss, na_22, label="Na-22", p0=p0_na)

#generate test data for skew function
num_points = 100
data_test = np.zeros((num_points, 2))
data_test[:, 0] = np.random.rand(num_points) * 10
data_test[:, 1] = gauss(data_test[:, 0], 5, 1) + np.random.rand(num_points) * 0.01


#test skew fit
p0_gauss = [1, 1]
big_fit(gauss, data_test, label="Gauss Test", p0=p0_gauss)


#generate test data for skew function
num_points = 1000
data_test = np.zeros((num_points, 2))
data_test[:, 0] = np.random.rand(num_points) * 10
data_test[:, 1] = skew(data_test[:, 0], 2, 1, 10)

#test skew fit
p0_skew = [2, 10, 9]
big_fit(skew, data_test, label="Skew Test", p0=p0_skew)












