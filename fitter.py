from scipy import optimize
from scipy.stats.distributions import t as t_stat
from scipy.stats import chisquare

import matplotlib
matplotlib.use('Agg')

matplotlib.rcParams['figure.facecolor'] = 'white'
import matplotlib.pyplot as plt
import numpy as np
import math as m
import sys, os

parameters = ["A", "V", "Y_0", "T_0", "\epsilon"]

def lf(f):
    float_str = "{0:.6g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def JD_to_MJD(jd):
    return jd - 2400000.5


def MJD_to_JD(mjd):
    return mjd + 2400000.5


def magAB_flux_Jy(mag, mag_err):
    p = -0.4 * (mag - 8.9)
    flux = 10 ** p
    flux_err = -0.4 * flux * np.log(10) * mag_err
    return flux, np.abs(flux_err)


def flux_Jy_magAB(flux, flux_err):
    mag = -2.5 * np.log10(flux) + 8.9
    mag_err = (- 2.5 / np.log(10)) * flux_err / flux
    return mag, np.abs(mag_err)


def magI_flux_Jy(mag, mag_err):
    # http://astro.pas.rochester.edu/~aquillen/ast142/costanti.html
    # band: I
    # Eff wavelength: 8000A
    # m = 0 flux: 2241 Jy
    # m = 0 flux: 1.22e-9 erg/s/cm^2/Angstrom
    # extinction relative to V band: 0.48
    f = 2247 * (10 ** (-mag / 2.5))
    df = - f * m.log(10) * mag_err / 2.5
    return f, np.abs(df)


def flux_Jy_magI(flux, flux_err):
    # see magI_flux_Jy(...)
    mag = -2.5 * (np.log10(flux) - np.log10(2247))
    mag_err = (- 2.5 / np.log(10)) * flux_err / flux
    return mag, np.abs(mag_err)


def read_ogle_phot_data(fname, min_t_mjd, max_t_mjd):
    # used columns: JD time, I magnitude, I magnitude uncertanty
    data = np.loadtxt(fname, usecols=[0, 1, 2])
    t0 = JD_to_MJD(data[:, 0])
    mask = np.logical_and(t0 >= min_t_mjd, t0 <= max_t_mjd)
    mag0 = data[:, 1]
    dmag0 = data[:, 2]
    f0, df0 = magI_flux_Jy(mag0, dmag0)
    t = t0[mask]
    f, df = f0[mask], df0[mask]
    return t, f, df, t0, f0, df0


def K0(X, A, V, Y0, T0):  # analytical solution
    T = X - T0
    Y = np.sqrt(Y0 * Y0 + V * V * T * T)
    K = A * (Y * Y + 2) / (Y * np.sqrt(Y * Y + 4))
    return K


def K(X, A, V, Y0, T0, epsilon):
    R0 = 1
    a = 2 + epsilon
    T = X - T0
    Y = np.sqrt(Y0 * Y0 + V * V * T * T)
    K = []
    for y in Y:
        f = lambda r: np.abs(np.abs(r * (1 - np.abs(R0/r) ** a)) - y)
        r1 = optimize.fminbound(f, 1e-12, R0, full_output=True, xtol=1e-9, maxfun=10000, disp=False)
        r2 = optimize.fminbound(f, R0, 1e12,  full_output=True, xtol=1e-9, maxfun=10000, disp=False)
        r1 = r1[0]
        r2 = r2[0]     
        
        #f = lambda r: np.abs(r * (1 - np.abs(R0/r) ** a) - y)
        #r1 = optimize.fmin(f, 1e-12, full_output=True, xtol=1e-9, ftol=1e-9, maxiter=10000, maxfun=10000, disp=False)
        #r2 = optimize.fmin(f, R0 + 1e-12, full_output=True, xtol=1e-9, ftol=1e-9, maxiter=10000, maxfun=10000, disp=False)
        #r1 = r1[0][0]
        #r2 = r2[0][0]
        K1 = 1 / np.abs(D(r1, R0, a))
        K2 = 1 / np.abs(D(r2, R0, a))
        K.append(K1 + K2)
    return A * np.asarray(K)


def D(r, R0, a):
    tmp = np.abs(R0 / r) ** a
    return (1 - tmp) * (1 + (a - 1) * tmp)


def get_event_data(filename, Tmax_JD, tau):
    Tmax_MJD = JD_to_MJD(float(Tmax_JD))
    tau = float(tau)
    tmin = Tmax_MJD - (3 * tau)
    tmax = Tmax_MJD + (3 * tau)
    print("Reading event data: %s" % filename)
    print("Tmax: %s (MJD), tau: %s, t_min: %s, t_max: %s" % (Tmax_MJD, tau, tmin, tmax))
    x, y, sy, x0, y0, sy0 = read_ogle_phot_data(filename, tmin, tmax)
    file_min_t = np.min(x0)
    file_max_t = np.max(x0)
    if file_min_t > Tmax_MJD - tau and file_max_t < Tmax_MJD + tau:
        print("The maximum time is too close to beginning and to the end of the dataset")
        return None
    return np.column_stack([x, y, sy]), np.column_stack([x0, y0, sy0]), Tmax_MJD, tmin, tmax


def get_simple_data(filename, error_rate=0.01, tmin=None, tmax=None, error=None):
    data = np.loadtxt(filename)
    Tmax = data[:,0][np.argmax(data[:,1])]
    x, y, sy = data[:,0], data[:,1], error_rate*data[:,1] if error is None else error*np.ones_like(data[:,1])
    if tmin is None:
        tmin = np.min(x)
    if tmax is None:
        tmax = np.max(x)
    return np.column_stack([x, y, sy]), np.column_stack([x, y, sy]), Tmax, tmin, tmax


def process_event(data, Tmax_MJD, tmin, tmax):
    xdata, ydata, yerr = data[:, 0], data[:, 1], data[:, 2]
    pinit0 = np.array([np.max(ydata), 1.0, 1.0, Tmax_MJD])
    print("Starting data fitting with point-lens model, initial guess: %s" % pinit0.tolist())
    par0, cov0, jac0, msg0, ret0 = optimize.leastsq(lambda p: (ydata - K0(xdata, *p)) / yerr, pinit0, full_output=True,
                                                    maxfev=1000000)
    print("=> Result: %s" % par0.tolist())

    # pinit = np.array([*par0, 0.0])
    # pinit[0] = np.max(ydata)

    # bounds = [
    #    [-10*np.abs(i) for i in pinit[:-1]] + [-1],
    #    [+10*np.abs(i) for i in pinit[:-1]] + [+1]
    # ]

    # Dummy initial parameters and bound
    #A, V, Y0, T0, epsilon
    pinit = np.array([par0[0], 1.0, 1.0, par0[3], 0.0])
    bounds = [[0, 0, 0, tmin, -0.9999], [1e10, 1e10, 1e10, tmax, 0.9999]]

    print("Starting data fitting with epsilon-enabled-lens model, initial guess: %s" % pinit.tolist())
    print("Using bounds: %s" % bounds)
    # par, cov, jac, msg, ret = optimize.leastsq(lambda p: (ydata - K(xdata, *p))/yerr, pinit, full_output=True, maxfev=1000000)
    par, cov = optimize.curve_fit(K, xdata, ydata, p0=pinit, sigma=yerr, maxfev=2000, bounds=bounds, method='trf')
    # par, cov = optimize.curve_fit(K, xdata, ydata, p0=pinit, sigma=yerr, maxfev=2000, method='trf')
    jac, msg, ret = None, None, 1
    print("=> Result: %s" % par.tolist())

    if ret > 4 or ret <= 0:
        print("=== Failed to find solution ===")
        print("ier: %d" % ret)
        print(
            "description:\n################### Description ###################\n%s\n###################################################" % msg)
        sys.exit(-1)

    return par, cov, par0, cov0


def get_accuracies(pars, pcov, ydata):
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)

    n = len(ydata)  # number of data points
    p = len(pars)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t_stat.ppf(1.0 - alpha / 2., dof)

    sigmas = []

    for i, p, var in zip(range(n), pars, np.diag(pcov)):
        sigma = np.sqrt(var)
        sigmas.append(sigma * tval)

    return sigmas, dof


def save_results(output_file, event, pars, sigma, chisq, dof):
    write_header = not os.path.exists(output_file)
    with open(output_file, "a") as f:
        if write_header:
            f.write("#event\t\chi^2\tDOF\t%s\n" % "\t".join([("%s\t%s_err" % (parameters[i], parameters[i])) for i in range(len(pars))]))
        f.write("%s\t%s\t%s\t%s\n" % (event, chisq, dof, "\t".join([("%s\t%s" % (pars[i], sigma[i])) for i in range(len(pars))])))


def output_results(event, data, data0, pars, pcov, pars0, pcov0, tmin, tmax, output_folder, img_folder):
    event_lower = event.lower()
    xplot = np.linspace(tmin, tmax, int((tmax - tmin) * 10))
    xdata, ydata, yerr = data[:, 0], data[:, 1], data[:, 2]
    xdata0, ydata0, yerr0 = data0[:, 0], data0[:, 1], data0[:, 2]

    sigmas, dof = get_accuracies(pars, pcov, ydata)
    sigmas0, dof0 = get_accuracies(pars0, pcov0, ydata)

    chisq = np.sum(((ydata - K(xdata, *pars)) / yerr) ** 2)
    chisq0 = np.sum(((ydata - K0(xdata, *pars0)) / yerr) ** 2)
    
    label = ""
    for i in range(len(pars)):
        label += "$%s=%s\pm%s$\n" % (parameters[i], lf(pars[i]), lf(sigmas[i]))
    label += "$\chi^2=%s$, $DOF=%s$\n" % (lf(chisq), dof)

    label0 = ""
    for i in range(len(pars0)):
        label0 += "$%s=%s\pm%s$\n" % (parameters[i], lf(pars0[i]), lf(sigmas0[i]))
    label0 += "$\chi^2=%s$, $DOF=%s$" % (lf(chisq0), dof0)

    plt.clf()
    SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.', color='grey')  # Data
    plt.plot(xplot, K0(xplot, *pars0), dashes=[3, 2], label=label0)  # Fit
    plt.plot(xplot, K(xplot, *pars), dashes=[2, 1], label=label)  # Fit
    plt.title('Best fit for %s' % event)
    plt.xlabel('HJD - 2450000 (days)')
    plt.ylabel('Flux (mJy)')
    plt.legend()
    # leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True)
    # for item in leg.legendHandles:
    #    item.set_visible(False)

    plt.subplot(2, 1, 2)
    plt.errorbar(xdata0, ydata0, yerr=yerr0, fmt='k.', color='grey')  # Data
    plt.axvspan(tmin, tmax, color='red', alpha=0.5)
    plt.xlabel('HJD - 2450000 (days)')
    plt.ylabel('Flux (mJy)')
    # plt.show() ################################### for automatic processing
    plt.savefig(os.path.join(img_folder, "%s.png" % event_lower))

    save_results(os.path.join(output_folder, "fitting_parameters.dat"), event_lower, pars, sigmas, chisq, dof)
    save_results(os.path.join(output_folder, "fitting_parameters_point.dat"), event_lower, pars0, sigmas0, chisq0, dof0)


def process_simple_event(event='test', phot_filename="1.dat", output_folder="./", img_folder="./", verbose=False, tmin=None, tmax=None, error_rate=0.01, error=None):
    ret = get_simple_data(phot_filename, tmin=tmin, tmax=tmax, error_rate=error_rate, error=error)
    if ret is None:
        return -1
    data, data0, Tmax, tmin, tmax = ret
    print("Tmax=%s, tmin=%s, tmax=%s" % (Tmax, tmin, tmax))
    pars, pcov, pars0, pcov0 = process_event(data, Tmax, tmin, tmax)
    output_results(event, data, data0, pars, pcov, pars0, pcov0, tmin, tmax, output_folder, img_folder)
    if verbose:
        return event, data, data0, pars, pcov, pars0, pcov0, tmin, tmax
    return 0


def process_ogle_event(event_folder="./", output_folder="./", img_folder="./", coefficient=3, verbose=False):
    if not os.path.exists(event_folder) or not os.path.exists(output_folder) or not os.path.exists(img_folder):
        print("One of input or output folders does not exist, please verify!")
        return -2
    params_filename = os.path.join(event_folder, 'params.dat')
    phot_filename = os.path.join(event_folder, 'phot.dat')
    event = None
    Tmax = None
    tau = None
    with open(params_filename, "r") as f:
        for l in f.read().split('\n'):
            l = l.strip().split()
            if len(l) > 0 and l[0] == 'Tmax':
                Tmax = float(l[1])
            if len(l) > 0 and l[0] == 'tau':
                tau = float(l[1])
            if len(l) > 0 and l[0].startswith('OGLE'):
                event = l[0]
    ret = get_event_data(phot_filename, Tmax, coefficient*tau)
    if ret is None:
        return -1
    data, data0, Tmax, tmin, tmax = ret
    pars, pcov, pars0, pcov0  = process_event(data, Tmax, tmin, tmax)
    output_results(event, data, data0, pars, pcov, pars0, pcov0, tmin, tmax, output_folder, img_folder)
    if verbose:
        return event, data, data0, pars, pcov, pars0, pcov0, tmin, tmax
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s path_to_ogle_event output_folder image_output_folder" % sys.argv[0])
        sys.exit(-1)
    r = process_ogle_event(event_folder=sys.argv[1], output_folder=sys.argv[2], img_folder=sys.argv[3])
    sys.exit(r)
