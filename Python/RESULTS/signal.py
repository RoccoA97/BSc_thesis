import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.stats import chi2, chisquare
from scipy.special import erfinv

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

nbins= 8
patience = '5000'
fontsize=13
alpha_axvspan=0.3


def __plot_t_distribution__(t):
    N, bins, patches = plt.hist(t, bins=nbins, edgecolor='black', linewidth=2, density=True, label='$t$ distribution with signal events')

    dof = 96
    mean, var, skew, kurt = chi2.stats(dof, moments='mvsk')
    x = np.linspace(chi2.ppf(0.001,dof), chi2.ppf(0.999,dof), 1000)
    plt.plot(x, chi2.pdf(x,dof), 'r-', lw=2, alpha=0.6, label='$\chi^2$ '+str(dof)+' dof')

    median = np.median(np.array(t))
    p_med = chi2.cdf(median,dof)
    pp_med = 1 - p_med
    sigma_med = np.sqrt(2) * erfinv(1-pp_med)

    mean = np.mean(t)
    p_mean = chi2.cdf(mean,dof)
    pp_mean = 1 - p_mean
    sigma_mean = np.sqrt(2) * erfinv(1-pp_mean)

    textstr = "Number of toys:\n%d\nMedian Bkg only:\n%2.2f\nMedian significance:\n%2.2f$\sigma$" %(t.shape[0],
                                                                                round(np.median(np.array(t)),2),
                                                                                round(sigma_med,2))
    plt.annotate(textstr, xy=(0.68, 0.50), xycoords='axes fraction',
         #verticalalignment='top',horizontalalignment='right',
         fontsize=fontsize)#, bbox=props)

    plt.xlabel('t',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.ylim(0.0, 0.040)
    plt.title('Distribution of t',fontsize=fontsize)
    plt.legend(fontsize=fontsize)

def __plot_percentiles__(x, percentiles, quantiles, ymax1=300):
    plt.xlabel('Epoch [k]',fontsize=fontsize)
    plt.ylabel('t',fontsize=fontsize)
    plt.title('Percentile plot',fontsize=fontsize)
    plt.xlim(0, 350)
    plt.ylim(0, ymax1)

    legend=[]

    for j in range(percentiles.shape[1]):
        plt.plot(x, percentiles[:, j], marker='.')
        legend.append(str(quantiles[j])+' \% quantile')

    plt.axvspan(300, 350, facecolor='grey', alpha=alpha_axvspan)
    plt.legend(legend, loc=4, fontsize=10)
    plt.grid()


def __retrieve_name__(FILE_NAME):
    name = FILE_NAME.split('/')[-1]
    name = name[:-3]
    return name


def __analysis_plot__(FILE_NAME, OUTPUT_DIR_NAME):
    f = h5py.File(FILE_NAME)

    epochs = np.array(f.get('epochs'))
    t = np.array(f.get('t'))
    history = np.array(f.get('history'))
    quantiles = np.array(f.get('quantiles'))
    percentiles = np.array(f.get('percentiles'))
    chi2_values = np.array(f.get('chi2_values'))
    p_values = np.array(f.get('p_values'))

    x = epochs / 1000
    f.close()


    fig=plt.figure(figsize=(13, 4.5))

    plt.subplot(1,2,1)
    __plot_t_distribution__(t)
    plt.subplot(1,2,2)
    __plot_percentiles__(x, percentiles, quantiles)

    #plt.show()
    file = __retrieve_name__(FILE_NAME)

    if not os.path.exists('./Python/RESULTS/' + OUTPUT_DIR_NAME):
        os.makedirs('./Python/RESULTS/' + OUTPUT_DIR_NAME)

    fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.11,top = 0.92)
    fig.savefig('./Python/RESULTS/' + OUTPUT_DIR_NAME + '/' + file.replace('.','-') + '.pdf')
    plt.clf()
    fig.clf()


# __analysis_plot__('./Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/ref100000_bkg20000_sig0/data_ref100000_bkg20000_sig0_wclip2.15.h5')

for dirname in os.listdir('./Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results_sig/'):
    for filename in os.listdir('./Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results_sig/' + dirname):
        datafile = './Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results_sig/' + dirname + '/' + filename
        __analysis_plot__(datafile, dirname)
    print(dirname)
