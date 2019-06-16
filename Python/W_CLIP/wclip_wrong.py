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

    textstr = "Number of toys:\n%d\nMedian Bkg only:\n%2f\n$\chi^2$:\n%2f" %(t.shape[0],
                                                                            round(np.median(np.array(t)),2),
                                                                            round(chi2_test,2))
    plt.annotate(textstr, xy=(0.73, 0.50), xycoords='axes fraction',
         #verticalalignment='top',horizontalalignment='right',
         fontsize=fontsize)#, bbox=props)

    plt.xlabel('t',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.xlim(40, 160)
    plt.ylim(0.0, 0.040)
    plt.title('Distribution of t',fontsize=fontsize)
    plt.legend(fontsize=fontsize)



def __chi2_plot__(x, chi2_values, ymax=10000):
    plt.xlabel('Epoch [k]',fontsize=fontsize)
    plt.ylabel('$\chi^2$ observed',fontsize=fontsize)
    plt.title('$\chi^2$ observed during training',fontsize=fontsize)
    plt.yscale('log')
    plt.xlim(0, 350)
    plt.ylim(0, ymax)

    legend = []

    plt.plot(x[:], chi2_values[:], marker='.')
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



def __retrieve_name__(FILE_NAME):
    name = FILE_NAME.split('/')[-1]
    name = name[:-3]
    return name



def __analysis_plot__(FILE_NAME_1, FILE_NAME_2):
    DIR_1 = './Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/ref200000_bkg20000_sig0/'
    DIR_2 = './Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/ref1000000_bkg20000_sig0/'

    f1 = h5py.File(DIR_1 + FILE_NAME_1,'r')
    print([keys for keys in f1.keys()])
    epochs_1 = np.array(f1.get('epochs'))
    t_1 = np.array(f1.get('t'))
    history_1 = np.array(f1.get('history'))
    quantiles_1 = np.array(f1.get('quantiles'))
    percentiles_1 = np.array(f1.get('percentiles'))
    chi2_values_1 = np.array(f1.get('chi2_values'))
    p_values_1 = np.array(f1.get('p_values'))
    f1.close()

    f2 = h5py.File(DIR_2 + FILE_NAME_2,'r')
    print([keys for keys in f2.keys()])
    epochs_2 = np.array(f2.get('epochs'))
    t_2 = np.array(f2.get('t'))
    history_2 = np.array(f2.get('history'))
    quantiles_2 = np.array(f2.get('quantiles'))
    percentiles_2 = np.array(f2.get('percentiles'))
    chi2_values_2 = np.array(f2.get('chi2_values'))
    p_values_2 = np.array(f2.get('p_values'))
    f2.close()

    x_1 = epochs_1 / 1000
    x_2 = epochs_2 / 1000


    fig=plt.figure(figsize=(13, 9))

    plt.subplot(2,2,1)
    __plot_t_distribution__(t_1, chi2_values_1[-1], p_values_1[-1])
    plt.subplot(2,2,2)
    __plot_t_distribution__(t_2, chi2_values_2[-1], p_values_2[-1])
    plt.subplot(2,2,3)
    __chi2_plot__(x_1, chi2_values_1)
    plt.subplot(2,2,4)
    __chi2_plot__(x_2, chi2_values_2)

    fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.06,top = 0.96)
    fig.savefig('./Python/W_CLIP/' + 'wrong_wclip_2-2_3-0' + '.pdf')
    plt.clf()
    fig.clf()




__analysis_plot__('data_ref200000_bkg20000_sig0_wclip2.2.h5', 'data_ref1000000_bkg20000_sig0_wclip3.h5')
