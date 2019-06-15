import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import random
import datetime
import h5py


INPUT_PATH_BKG = './Z_5D_DATA/EFT_YW06/bkg/'
INPUT_PATH_SIG = './Z_5D_DATA/EFT_YW06/sig/'

N_Sig = 16000
N_Bkg = 1600000
nbins = 50


for i in range(150):
    if i==0:
        filename = 'EFT_YW06_bkg_' + str(i) + '.h5'
        f = h5py.File(INPUT_PATH_BKG + filename, "r")
        test_Bkg = np.array(f.get('HLF'))
        print(test_Bkg)
        test_All_Bkg = test_Bkg
        f.close()
    if i!=0:
        filename = 'EFT_YW06_bkg_' + str(i) + '.h5'
        f = h5py.File(INPUT_PATH_BKG + filename, "r")
        test_Bkg = np.array(f.get('HLF'))
        test_All_Bkg = np.concatenate((test_All_Bkg,test_Bkg), axis=0)
        f.close()

for i in range(50):
    if i==0:
        filename = 'EFT_YW06_sig_' + str(i) + '.h5'
        f = h5py.File(INPUT_PATH_SIG + filename, "r")
        test_Sig = np.array(f.get('HLF'))
        test_All_Sig = test_Sig
        f.close()
    if i!=0:
        filename = 'EFT_YW06_sig_' + str(i) + '.h5'
        f = h5py.File(INPUT_PATH_SIG + filename, "r")
        test_Sig = np.array(f.get('HLF'))
        test_All_Sig = np.concatenate((test_All_Sig,test_Sig), axis=0)
        f.close()


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

pt_bins = np.concatenate((np.arange(0,250,15), np.arange(250,500,50)), axis=0)
pt_bins = np.concatenate((pt_bins, np.arange(500,1000,100)), axis=0)
mass_bins = np.concatenate((np.arange(0,250,10), np.arange(250,500,50)), axis=0)
mass_bins = np.concatenate((mass_bins, np.arange(500,1000,100)), axis=0)

fig = plt.figure(figsize=(13, 18))
plt.clf()
fig.clf()

plt.subplot(3,2,1)
plt.title('$p_{T,1}$ distribution')
plt.xlabel('$p_{T,1}$ [GeV/c]')
plt.ylabel('Density')
plt.yscale('log')
plt.hist(test_All_Bkg[:,0], bins=pt_bins, range=(0,1000), histtype='step', linewidth=2, density=True, label='Background')
plt.hist(test_All_Sig[:,0], bins=pt_bins, range=(0,1000), histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()

plt.subplot(3,2,2)
plt.title('$p_{T,2}$ distribution')
plt.xlabel('$p_{T,2}$ [GeV/c]')
plt.ylabel('Density')
plt.yscale('log')
plt.hist(test_All_Bkg[:,1], bins=pt_bins, range=(0,1000), histtype='step', linewidth=2, density=True, label='Background')
plt.hist(test_All_Sig[:,1], bins=pt_bins, range=(0,1000), histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()

plt.subplot(3,2,3)
plt.title('$\eta_{1}$ distribution')
plt.xlabel('$\eta_{1}$')
plt.ylabel('Density')
plt.hist(test_All_Bkg[:,2], bins=100, histtype='step', linewidth=2, density=True, label='Background')
plt.hist(test_All_Sig[:,2], bins=100, histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()

plt.subplot(3,2,4)
plt.title('$\eta_{2}$ distribution')
plt.xlabel('$\eta_{2}$')
plt.ylabel('Density')
plt.hist(test_All_Bkg[:,3], bins=100, histtype='step', linewidth=2, density=True, label='Background')
plt.hist(test_All_Sig[:,3], bins=100, histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()

plt.subplot(3,2,5)
plt.title('$\Delta \phi$ distribution')
plt.xlabel('$\Delta \phi$ [rad]')
plt.ylabel('Density')
plt.hist(test_All_Bkg[:,4], bins=100, histtype='step', linewidth=2, density=True, label='Background')
plt.hist(test_All_Sig[:,4], bins=100, histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()

plt.subplot(3,2,6)
plt.title('$M_{Z}$ distribution')
plt.xlabel('$M_{Z}$ [Gev/c$^2$]')
plt.ylabel('Density')
plt.yscale('log')
plt.hist(test_All_Bkg[:,5], bins=mass_bins, range=(0,500), histtype='step', linewidth=2, density=True, label='Background')
plt.hist(test_All_Sig[:,5], bins=mass_bins, range=(0,500), histtype='step', linewidth=2, density=True, label='Signal')
plt.legend()

fig.subplots_adjust(left = 0.05,right = 0.95,bottom = 0.025,top = 0.975)
fig.savefig('./Python/Z/features_EFT_YW06.pdf')

plt.show()
plt.clf()
fig.clf()
