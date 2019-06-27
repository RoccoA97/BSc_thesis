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

y_Zprime = [2.15, 2.40, 2.45, 2.55, 2.70]
y_Zeft   = [2.25, 2.40, 2.45, 2.55, 2.70]

x_Zprime = np.array(x_Zprime)
x_Zeft   = np.array(x_Zeft)
y_Zprime = np.array(y_Zprime)
y_Zeft   = np.array(y_Zeft)

x_Zprime = x_Zprime / N_Bkg
x_Zeft   = x_Zeft / N_Bkg



fig=plt.figure(figsize=(13, 5))

plt.subplot(1,2,1)
plt.title('Weight clipping curve - \\textbf{Zprime-Zmumu}', fontsize=fontsize)
plt.xlabel('$\\frac{N_{\\mathrm{ref}}}{N_{\\mathrm{bkg}}}$', fontsize=fontsize)
plt.ylabel('$W$', fontsize=fontsize)
plt.ylim(2.1,2.9)
plt.plot(x_Zprime, y_Zprime, 'C0o-')

plt.subplot(1,2,2)
plt.title('Weight clipping curve - \\textbf{EFT\_YW06}', fontsize=fontsize)
plt.xlabel('$\\frac{N_{\\mathrm{ref}}}{N_{\\mathrm{bkg}}}$', fontsize=fontsize)
plt.ylabel('$W$', fontsize=fontsize)
plt.ylim(2.1,2.9)
plt.plot(x_Zeft, y_Zeft, 'C1o-')

plt.show()
fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.11,top = 0.93)
fig.savefig('./Python/W_CLIP/W_CLIP_CURVE/w_clip_curve.pdf')
plt.clf()
fig.clf()
