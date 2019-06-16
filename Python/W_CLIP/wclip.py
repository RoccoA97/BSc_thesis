import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.stats import chi2, chisquare

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

nbins= 8
xmin = 60.0
xmax = 140.0
patience = '5000'
fontsize=13
alpha_axvspan=0.3


def __plot_t_distribution__(t, chi2_test, p_val):
    N, bins, patches = plt.hist(t, bins=nbins, range=[xmin,xmax], edgecolor='black', linewidth=2, density=True, label='Observed $t$ distribution')

    dof = 96
    mean, var, skew, kurt = chi2.stats(dof, moments='mvsk')
    x = np.linspace(chi2.ppf(0.001,dof), chi2.ppf(0.999,dof), 1000)
    plt.plot(x, chi2.pdf(x,dof), 'r-', lw=2, alpha=0.6, label='$\chi^2$ '+str(dof)+' dof')

    textstr = "Number of toys:\n%d\nMedian Bkg only:\n%2f\n$\chi^2$:\n%2f\n$\chi^2$ test $p$-value:\n%2f" %(t.shape[0],
                                                                                round(np.median(np.array(t)),2),
                                                                                round(chi2_test,2),
                                                                                round(p_val, 3))
    plt.annotate(textstr, xy=(0.73, 0.40), xycoords='axes fraction',
         #verticalalignment='top',horizontalalignment='right',
         fontsize=fontsize)#, bbox=props)

    plt.xlabel('t',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.xlim(40, 160)
    plt.ylim(0.0, 0.040)
    plt.title('Distribution of t',fontsize=fontsize)
    plt.legend(fontsize=fontsize)

def __plot_percentiles__(x, percentiles, quantiles, ymax1=150):
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
    plt.legend(legend, loc=4, fontsize=fontsize)
    plt.grid()

def __chi2_plot__(x, chi2_values, ymax=50):
    plt.xlabel('Epoch [k]',fontsize=fontsize)
    plt.ylabel('$\chi^2$ observed',fontsize=fontsize)
    plt.title('$\chi^2$ observed during training',fontsize=fontsize)
    plt.xlim(0, 350)
    plt.ylim(0, ymax)

    legend = []

    plt.plot(x[10:], chi2_values[10:], marker='.')
    legend.append('$\chi^2$ observed')
    plt.plot([0, 350],[nbins-1, nbins-1])
    legend.append('dof = 7')
    plt.plot([0, 350],[nbins-1+np.sqrt(2*(nbins-1)), nbins-1+np.sqrt(2*(nbins-1))])
    legend.append('dof + $\sigma$')
    plt.plot([0, 350],[nbins-1+2*np.sqrt(2*(nbins-1)), nbins-1+2*np.sqrt(2*(nbins-1))])
    legend.append('dof + 2$\sigma$')
    plt.plot([0, 350],[nbins-1+3*np.sqrt(2*(nbins-1)), nbins-1+3*np.sqrt(2*(nbins-1))])
    legend.append('dof + 3$\sigma$')
    plt.axvspan(300, 350, facecolor='grey', alpha=alpha_axvspan)

    plt.legend(legend, loc=1, fontsize=fontsize)
    plt.grid()

def __p_value_plot__(x, p_values, ymax=1.0):
    plt.xlabel('Epoch [k]',fontsize=fontsize)
    plt.ylabel('$p$-value observed',fontsize=fontsize)
    plt.title('$p$-value observed during training',fontsize=fontsize)
    plt.xlim(0, 350)
    plt.ylim(0, ymax)

    legend = []

    plt.plot(x, p_values, marker='.')
    legend.append('$p$-value observed')

    plt.axvspan(300, 350, facecolor='grey', alpha=alpha_axvspan)
    plt.legend(legend, loc=1, fontsize=13)
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


    fig=plt.figure(figsize=(13, 9))

    plt.subplot(2,2,1)
    __plot_t_distribution__(t, chi2_values[-1], p_values[-1])
    plt.subplot(2,2,2)
    __plot_percentiles__(x, percentiles, quantiles)
    plt.subplot(2,2,3)
    __chi2_plot__(x, chi2_values)
    plt.subplot(2,2,4)
    __p_value_plot__(x, p_values)

    #plt.show()
    file = __retrieve_name__(FILE_NAME)

    if not os.path.exists('./Python/W_CLIP/' + OUTPUT_DIR_NAME):
        os.makedirs('./Python/W_CLIP/' + OUTPUT_DIR_NAME)

    fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.06,top = 0.96)
    fig.savefig('./Python/W_CLIP/' + OUTPUT_DIR_NAME + '/' + file.replace('.','-') + '.pdf')
    plt.clf()
    fig.clf()


# __analysis_plot__('./Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/ref100000_bkg20000_sig0/data_ref100000_bkg20000_sig0_wclip2.15.h5')

for dirname in os.listdir('./Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/'):
    for filename in os.listdir('./Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/' + dirname):
        datafile = './Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/' + dirname + '/' + filename
        __analysis_plot__(datafile, dirname)
    print(dirname)
