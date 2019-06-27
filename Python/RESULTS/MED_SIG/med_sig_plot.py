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

fontsize=13
N_Bkg = 20

x_Zprime = [100, 200, 300, 500, 1000]
x_Zeft   = [100, 200, 300, 500, 1000]

y_Zprime = [3.873, 4.025, 3.766, 3.721, 4.059]
y_Zeft   = [1.556, 1.216, 1.052, 1.100, 2.016]

x_Zprime = np.array(x_Zprime)
x_Zeft   = np.array(x_Zeft)
y_Zprime = np.array(y_Zprime)
y_Zeft   = np.array(y_Zeft)

x_Zprime = x_Zprime / N_Bkg
x_Zeft   = x_Zeft / N_Bkg


fig=plt.figure(figsize=(13, 5))

plt.subplot(1,2,1)
plt.title('Median $\sigma_{\mathrm{obs}}$ - \\textbf{Zprime-Zmumu}', fontsize=fontsize)
plt.xlabel('$\\frac{N_{\\mathrm{ref}}}{N_{\\mathrm{bkg}}}$', fontsize=fontsize)
plt.ylabel('$\sigma_{\mathrm{obs}}$', fontsize=fontsize)
# plt.ylim(2.1,2.9)
plt.plot(x_Zprime, y_Zprime, 'C0o-')

plt.subplot(1,2,2)
plt.title('Median $\sigma_{\mathrm{obs}}$ - \\textbf{EFT\_YW06}', fontsize=fontsize)
plt.xlabel('$\\frac{N_{\\mathrm{ref}}}{N_{\\mathrm{eft}}}$', fontsize=fontsize)
plt.ylabel('$\sigma_{\mathrm{obs}}$', fontsize=fontsize)
# plt.ylim(2.1,2.9)
plt.plot(x_Zeft, y_Zeft, 'C1o-')

plt.show()
fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.11,top = 0.93)
fig.savefig('./Python/RESULTS/MED_SIG/med_sig_plot.pdf')
plt.clf()
fig.clf()
