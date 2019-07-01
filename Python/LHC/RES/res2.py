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



def bkg(x):
    return -1.0*x*x*(0.15*x-0.5)

def sig(x):
    return 0.02/((x*x-0.8)**2 + 0.03)

xmin = 0
xmax = 3.332
samples = 2000000

x_bkg = []
x_sig = []

for i in range(samples):
    r = np.random.uniform(xmin,xmax)
    s = np.random.uniform(0,1)

    if s <= bkg(r):
        x_bkg.append(r)

    if s <= sig(r):
        x_sig.append(r)

plt.hist(x_bkg,bins=100)
plt.hist(x_sig,bins=100)
plt.show()
plt.close()
plt.clf()

x_all = x_bkg + x_sig
fig,ax = plt.subplots(1)
plt.hist(x_all,bins=100,density=True,histtype='step')
plt.ylim(0,1)

ax.set_yticklabels([])
ax.set_xticklabels([])
for tic in ax.xaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
for tic in ax.yaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False

plt.title('Resonance in the mass distribution', fontsize=fontsize)
plt.xlabel('$M_{cd}$', fontsize=fontsize)
plt.ylabel('$n(M_{cd})$', fontsize=fontsize)
#plt.tick_params(axis='both', labelsize=0, length = 0)

fig.subplots_adjust(left = 0.08,right = 0.92,bottom = 0.08,top = 0.92)
fig.savefig('./Python/LHC/RES/res2.pdf')
plt.show()
plt.clf()
