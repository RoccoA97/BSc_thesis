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

    textstr = "Number of toys:\n%d\nMedian Bkg only:\n%2.2f\nMedian significance:\n%2.3f$\sigma$" %(t.shape[0],
                                                                                round(np.median(np.array(t)),2),
                                                                                round(sigma_med,3))
    plt.annotate(textstr, xy=(0.68, 0.45), xycoords='axes fraction',
         #verticalalignment='top',horizontalalignment='right',
         fontsize=fontsize)#, bbox=props)

    plt.xlabel('t',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.ylim(0.0, 0.040)
    plt.title('Distribution of t',fontsize=fontsize)
    plt.legend(fontsize=fontsize)

def __plot_percentiles__(x, data, quantiles, ymax1=300):

    N_toys = data.shape[0]
    N_loss = data.shape[1]

    percentiles=np.array([])
    x = int(5000) * np.arange(N_loss) / 1000

    for j in range(data.shape[1]):
        percentiles_i = np.percentile(data[:, j], quantiles)
        percentiles_i = np.expand_dims(percentiles_i, axis=1)

        if j==0:
            percentiles = percentiles_i.T
        else:
            percentiles = np.concatenate((percentiles, percentiles_i.T))



    plt.xlabel('Epoch [k]',fontsize=fontsize)
    plt.ylabel('t',fontsize=fontsize)
    plt.title('Percentile plot',fontsize=fontsize)
    plt.xlim(0, 350)
    plt.ylim(0, percentiles[-1,4]+15)

    legend=[]

    for j in range(percentiles.shape[1]):
        plt.plot(x, percentiles[:, j], marker='.')
        legend.append(str(quantiles[j])+' \% quantile')

    plt.axvspan(300, 350, facecolor='grey', alpha=alpha_axvspan)
    plt.legend(legend, loc=4, fontsize=10)
    plt.grid()




def __analysis_plot__(DIR_NAME):
    epochs = range(0,305,5)

    history = []
    for i in range(100):
        filename = 'Toy5D_patience5000_1000000ref_20000data_'+str(i)+'_t.txt'
        if os.path.exists(DIR_NAME + filename):
            fp = open(DIR_NAME+filename)
            t = np.array([float(fp.readlines()[0])])
            print(t)
            fp.close()

        filename = 'Toy5D_patience5000_1000000ref_20000data_'+str(i)+'_history5000.h5'
        if os.path.exists(DIR_NAME + filename):
            print(DIR_NAME + filename)
            f = h5py.File(DIR_NAME+filename)
            loss = -2 * np.array(f.get('loss'))
            loss = np.concatenate((loss,t),axis=0)
            f.close()

            history.append(loss)

    history = np.array(history)
    t = history[:,-1]

    print(history.shape)
    print(t.shape)

    fig=plt.figure(figsize=(13, 4))

    plt.subplot(1,2,1)
    __plot_t_distribution__(t)
    plt.subplot(1,2,2)
    __plot_percentiles__(epochs, history, [2.5, 25, 50, 75, 97.5])


    if not os.path.exists('./Python/RESULTS/' + 'ref1000000_bkg0_eft20000'):
        os.makedirs('./Python/RESULTS/' + 'ref1000000_bkg0_eft20000')

    fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.12,top = 0.93)
    fig.savefig('./Python/RESULTS/' + 'ref1000000_bkg0_eft20000' + '/' + 'data_ref1000000_bkg0_eft20000_wclip2-7' + '.pdf')
    plt.clf()
    fig.clf()


# __analysis_plot__('./Z_5D_SIMULATIONS_RESULTS-master/wcliptest/results/ref100000_bkg20000_sig0/data_ref100000_bkg20000_sig0_wclip2.15.h5')
# C:\Users\Ardino Pierfrancesco\Desktop\Z_5D_SIMULATIONS_RESULTS-master\wcliptest\ref1000000_bkg0_eft20000\Z_5D_EFT_patience5000_ref1000000_data20000_epochs300000_latent5_layers3_wclip2.7
# C:\Users\Ardino Pierfrancesco\Desktop\Z_5D_SIMULATIONS_RESULTS-master\wcliptest\ref1000000_bkg0_eft20000\Z_5D_EFT_patience5000_ref1000000_data20000_epochs300000_latent5_layers3_wclip2.7
# Toy5D_patience5000_1000000ref_20000data_0_history5000
print(os.getcwd())
__analysis_plot__('C:\\Users\\Ardino Pierfrancesco\\Desktop\\Z_5D_SIMULATIONS_RESULTS-master\\wcliptest\\ref1000000_bkg0_eft20000\\Z_5D_EFT_patience5000_ref1000000_data20000_epochs300000_latent5_layers3_wclip2.7\\')
