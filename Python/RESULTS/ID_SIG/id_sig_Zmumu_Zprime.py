# ******************************************************************************
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import leastsq, curve_fit
from math import exp, cos, log, pi, sqrt
import sys
import os
import random
import datetime
import h5py
# ******************************************************************************

# ******************************************************************************
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fontsize=13
# ******************************************************************************

# ******************************************************************************
def PowerLaw(x,a,b,c,d):
    return a*np.exp(-b*x + c) + d

def DoubleGaussian(x,a,b,c,d,e,f):
    return a*np.exp(-((x-b)/c)**2) + d*np.exp(-((x-e)/f)**2)
# ******************************************************************************


# ******************************************************************************
N_Bkg = 1600000
N_Sig = 16000

N_bkg_toy = 20000
N_sig_toy = 40
# ******************************************************************************

# ******************************************************************************
INPUT_PATH_BKG = './Z_5D_DATA/Zmumu_lepFilter_13TeV/'
INPUT_PATH_SIG = './Z_5D_DATA/Zprime_lepFilter_13TeV/M300/'
FILE_ID_BKG = 'Zmumu_13TeV_20PU_'
FILE_ID_SIG = 'Zprime_lepFilter_300GeV_13TeV_'

seed=datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)

#random integer to select Zprime file between 0 and 9 (10 input files)
index_sig = np.arange(300)
index_sig = np.delete(index_sig,133,0)
np.random.shuffle(index_sig)
#random integer to select Zmumu file between 0 and 999 (1000 input files)
index_bkg = np.arange(1000)
np.random.shuffle(index_bkg)
# ******************************************************************************

# ******************************************************************************
HLF_REF = np.array([])
HLF_name = ''
i=0
for v_i in index_bkg:
    f = h5py.File(INPUT_PATH_BKG + FILE_ID_BKG + str(v_i) + '.h5')
    hlf = np.array(f.get('HLF'))
    #print(hlf.shape)
    hlf_names = f.get('HLF_names')
    if not hlf_names:
        continue
    #select the column with px1, pz1, px2, py2, pz2
    #cols = [0, 1, 7, 8, 9, 30]
    cols = [30]
    if i==0:
        HLF_REF=hlf[:, cols]
        i=i+1
    else:
        HLF_REF=np.concatenate((HLF_REF, hlf[:, cols]), axis=0)
    f.close()
    #print(HLF_REF.shape)
    if HLF_REF.shape[0]>=N_Bkg:
        HLF_REF = HLF_REF[:N_Bkg, :]
        break
print('HLF_REF+BKG shape')
print(HLF_REF.shape)

HLF_SIG = np.array([])
HLF_SIG_name = ''
i=0
for u_i in index_sig:
    f = h5py.File(INPUT_PATH_SIG + FILE_ID_SIG + str(u_i) + '.h5')
    hlf = np.array(f.get('HLF'))
    hlf_names = f.get('HLF_names')
    if not hlf_names:
        continue
    #select the column with px1, pz1, px2, py2, pz2
    #cols = [0, 1, 7, 8, 9, 30]
    cols = [30]
    if i==0:
        HLF_SIG=hlf[:, cols]
        i=i+1
    else:
        HLF_SIG=np.concatenate((HLF_SIG, hlf[:, cols]), axis=0)
    f.close()
    if HLF_SIG.shape[0]>=N_Sig :
        HLF_SIG=HLF_SIG[:N_Sig, :]
        break
print('HLF_SIG shape')
print(HLF_SIG.shape)
# ******************************************************************************

# ******************************************************************************
feature = np.concatenate((HLF_REF, HLF_SIG), axis=0)

background = feature[:N_Bkg,:]
signal = feature[N_Bkg:N_Bkg+N_Sig,:]
bkg_sig = feature[:N_Bkg+N_Sig,:]
# ******************************************************************************

# ******************************************************************************
print('M_Z distribution - background')

nbins_bkg = 100

plt.clf()
plt.title('$M_{Z}$ distribution')
plt.xlabel('$M_{Z}$')
plt.ylabel('Density')
plt.yscale('log')
Y_bkg, bins_bkg, patches_bkg = plt.hist(background[:,0],
                                        bins=nbins_bkg,
                                        range=(0,600),
                                        histtype='step',
                                        linewidth=2,
                                        density=False,
                                        label='Background')
# plt.hist(signal[:,0], bins=nbins, histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()
plt.show()
# ******************************************************************************

# ******************************************************************************
print('M_Z distribution - signal')

nbins_sig = 50

plt.clf()
plt.title('$M_{Z}$ distribution')
plt.xlabel('$M_{Z}$')
plt.ylabel('Density')
# plt.yscale('log')
Y_sig, bins_sig, patches_sig = plt.hist(signal[:,0],
                                        bins=nbins_sig,
                                        range=(200,400),
                                        histtype='step',
                                        linewidth=2,
                                        density=False,
                                        label='Signal')
# plt.hist(signal[:,0], bins=nbins, histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()
plt.show()
# ******************************************************************************

# ******************************************************************************
rescale = 1000
xmin = 250/rescale
xmax = 350/rescale

x_bkg = []
y_bkg = []

x_sig = []
y_sig = []

for i in range(len(bins_bkg)-1):
    if ((bins_bkg[i+1] + bins_bkg[i])/(2*rescale) > xmin) and ((bins_bkg[i+1] + bins_bkg[i])/(2*rescale) < xmax):
        x_bkg.append( (bins_bkg[i+1] + bins_bkg[i])/(2*rescale) )
        y_bkg.append( Y_bkg[i] )

for i in range(len(bins_sig)-1):
    if ((bins_sig[i+1] + bins_sig[i])/(2*rescale) > xmin) and ((bins_sig[i+1] + bins_sig[i])/(2*rescale) < xmax):
        x_sig.append( (bins_sig[i+1] + bins_sig[i])/(2*rescale) )
        y_sig.append( Y_sig[i] )

print(len(x_bkg))
print(len(y_bkg))

print(len(x_sig))
print(len(y_sig))

x_bkg = np.array(x_bkg)
y_bkg = np.array(y_bkg)

x_sig = np.array(x_sig)
y_sig = np.array(y_sig)
# ******************************************************************************

# ******************************************************************************
out_bkg, pcov_bkg = curve_fit(PowerLaw, x_bkg, y_bkg)
out_sig, pcov_sig = curve_fit(DoubleGaussian, x_sig, y_sig,
                             [2.09827877e+03, 2.99735747e-01, 2.78332413e-03, 2.80388489e+02, 2.97076315e-01, -1.00825202e-02])

pow_x_bkg = []
pow_y_bkg = []

brw_x_sig = []
brw_y_sig = []

for i in np.arange(xmin,xmax,0.001):
    pow_x_bkg.append(i)
    pow_y_bkg.append(PowerLaw(i,out_bkg[0],out_bkg[1],out_bkg[2],out_bkg[3]))

    brw_x_sig.append(i)
    brw_y_sig.append(DoubleGaussian(i,out_sig[0],out_sig[1],out_sig[2],out_sig[3],out_sig[4],out_sig[5]))
# ******************************************************************************

# ******************************************************************************
print('Fit - background')
plt.clf()
plt.plot(x_bkg,y_bkg)
plt.plot(pow_x_bkg,pow_y_bkg)
plt.show()

print(out_bkg[0],out_bkg[1],out_bkg[2],out_bkg[3])
# ******************************************************************************

# ******************************************************************************
print('Fit - signal')
plt.clf()
plt.plot(x_sig,y_sig)
plt.plot(brw_x_sig,brw_y_sig)
plt.show()
# ******************************************************************************

# ******************************************************************************
def sig_bkg(x,N_bkg_toy,N_sig_toy):
    bkg = (N_bkg_toy/N_Bkg)*PowerLaw(x,out_bkg[0],out_bkg[1],out_bkg[2],out_bkg[3])
    sig = (N_sig_toy/N_Sig)*DoubleGaussian(x,out_sig[0],out_sig[1],out_sig[2],out_sig[3],out_sig[4],out_sig[5])
    return bkg + sig
# ******************************************************************************

# ******************************************************************************
x_sig_bkg = []
y_sig_bkg = []

for i in np.arange(xmin,xmax,0.001):
    x_sig_bkg.append(i)
    y_sig_bkg.append(sig_bkg(i,N_bkg_toy,N_sig_toy)/(N_bkg_toy+N_sig_toy))
# ******************************************************************************

# ******************************************************************************
print('Fit - signal+background')

plt.clf()
plt.plot(x_sig_bkg,y_sig_bkg)
plt.show()
# ******************************************************************************

# ******************************************************************************
def log_ratio(x,N_bkg_toy,N_sig_toy):
    sig_p_bkg = sig_bkg(x,N_bkg_toy,N_sig_toy)
    bkg = (N_bkg_toy/N_Bkg)*PowerLaw(x,out_bkg[0],out_bkg[1],out_bkg[2],out_bkg[3])
    return log((sig_p_bkg/bkg)*(N_bkg_toy/(N_bkg_toy+N_sig_toy)))
# ******************************************************************************

# ******************************************************************************
x_log_ratio = []
y_log_ratio = []

for i in np.arange(xmin,xmax,0.001):
    x_log_ratio.append(i)
    y_log_ratio.append(log_ratio(i,N_bkg_toy,N_sig_toy))
# ******************************************************************************

# ******************************************************************************
print('Log-likelihood ratio')

plt.clf()
plt.plot(x_log_ratio,y_log_ratio)
plt.show()
# ******************************************************************************

# ******************************************************************************
integral = quad(log_ratio, xmin, xmax, args=(N_bkg_toy,N_sig_toy))[0]
print(integral)
print(sqrt(2*integral*1000))
# ******************************************************************************

# ******************************************************************************
x_bkg = np.array(x_bkg)
y_bkg = np.array(y_bkg)
pow_x_bkg = np.array(pow_x_bkg)
pow_y_bkg = np.array(pow_y_bkg)

x_sig = np.array(x_sig)
y_sig = np.array(y_sig)
brw_x_sig = np.array(brw_x_sig)
brw_y_sig = np.array(brw_y_sig)

x_sig_bkg = np.array(x_sig_bkg)
y_sig_bkg = np.array(y_sig_bkg)

x_log_ratio = np.array(x_log_ratio)
y_log_ratio = np.array(y_log_ratio)
# ******************************************************************************

# ******************************************************************************
fig=plt.figure(figsize=(13, 11))
plt.clf()

plt.subplot(2,2,1)
plt.title('$M_{Z}$ distribution - background', fontsize=fontsize)
plt.xlabel('$M_{Z}$ [GeV/c$^2$]', fontsize=fontsize)
plt.ylabel('$n_{\mathrm{bkg}}$', fontsize=fontsize)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.plot(x_bkg*1000,y_bkg/N_Bkg, label='Data')
plt.plot(pow_x_bkg*1000,pow_y_bkg/N_Bkg, label='Fit')
plt.legend(loc=1, fontsize=fontsize)

plt.subplot(2,2,2)
plt.title('$M_{Z}$ distribution - signal', fontsize=fontsize)
plt.xlabel('$M_{Z}$ [GeV/c$^2$]', fontsize=fontsize)
plt.ylabel('$n_{\mathrm{sig}}$', fontsize=fontsize)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.plot(x_sig*1000,y_sig/N_Bkg, label='Data')
plt.plot(brw_x_sig*1000,brw_y_sig/N_Bkg, label='Fit')
plt.legend(loc=1, fontsize=fontsize)

plt.subplot(2,2,3)
plt.title('$M_{Z}$ distribution - signal + background', fontsize=fontsize)
plt.xlabel('$M_{Z}$ [GeV/c$^2$]', fontsize=fontsize)
plt.ylabel('$n_{\mathrm{sig+bkg}}$', fontsize=fontsize)
plt.yscale('log')
plt.plot(x_sig_bkg*1000,y_sig_bkg, label='Fit')
plt.legend(loc=1, fontsize=fontsize)

plt.subplot(2,2,4)
plt.title('Log-ratio of $n_{\mathrm{sig+bkg}}$ and $n_{\mathrm{bkg}}$', fontsize=fontsize)
plt.xlabel('$M_{Z}$ [GeV/c$^2$]', fontsize=fontsize)
plt.ylabel('Log-ratio', fontsize=fontsize)
plt.plot(x_log_ratio*1000,y_log_ratio, label='Log-ratio')
plt.legend(loc=1, fontsize=fontsize)

fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.06,top = 0.96,hspace=0.25)
fig.savefig('./Python/RESULTS/ID_SIG/id_sig_Zmumu_Zprime.pdf')
plt.show()
plt.clf()
fig.clf()
# ******************************************************************************
