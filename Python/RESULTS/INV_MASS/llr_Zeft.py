import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.stats import chi2, chisquare
from scipy.special import erfinv


toys = [3, 6, 9, 12,
        15, 18, 21, 24,
        27, 30, 33, 36,
        39, 42, 45, 48]
verbose = 1
fontsize = 13

print(len(toys))
FILE_PATH = 'C:\\Users\\Ardino Pierfrancesco\\Desktop\\Z_5D_SIMULATIONS_RESULTS-master\\wcliptest\\ref300000_bkg0_eft20000\\epochs300000\\Z_5D_patience5000_ref300000_bkg0_eft20000_epochs300000_latent5_wclip2.45_run1\\'
INPUT_PATH_BKG = './Z_5D_DATA/EFT_YW06/bkg/'
INPUT_PATH_SIG = './Z_5D_DATA/EFT_YW06/sig/'



def __read_from_h5_file_predictions__(FILE_PATH, toys):
    """Read .h5 files to retrieve loss functions values for every toy during training
    """
    predictions = []
    prediction = []
    for filename in os.listdir(FILE_PATH):
        for i in toys:
            if filename != 'all.txt':
                if filename.split('_')[-2] == str(i):
                    if filename.split('_')[-1] == ('predictions.h5'):
                        print(filename)
                        f = h5py.File(FILE_PATH+filename)
                        feature = np.array(f.get('feature'))
                        seed = np.array(f.get('seed'))[()]
                        print(seed)
                        target = np.array(f.get('target'))
                        u = np.array(f.get('u'))
                        v = np.array(f.get('v'))
                        mass = np.array(f.get('mass'))

                        N_ref = feature.shape[0] - 20000
                        N_Bkg = 0
                        N_Eft = 20000

                        HLF_REF = np.array([])
                        HLF_name = ''
                        i=0
                        for v_i in v:
                            f = h5py.File(INPUT_PATH_BKG+'EFT_YW06_bkg_'+str(v_i)+'.h5')
                            hlf = np.array(f.get('HLF'))
                            #print(hlf.shape)
                            cols = [5]
                            if i==0:
                                HLF_REF=hlf[:, cols]
                                i=i+1
                            else:
                                HLF_REF=np.concatenate((HLF_REF, hlf[:, cols]), axis=0)
                            f.close()
                            #print(HLF_REF.shape)
                            if HLF_REF.shape[0]>=N_ref+N_Bkg:
                                HLF_REF = HLF_REF[:N_ref+N_Bkg, :]
                                break
                        print('HLF_REF+BKG shape')
                        print(HLF_REF.shape)



                        #SIGNAL
                        #extract N_Sig events from Zprime files
                        HLF_SIG = np.array([])
                        HLF_SIG_name = ''
                        i=0
                        for u_i in u:
                            f = h5py.File(INPUT_PATH_SIG+'EFT_YW06_sig_'+str(u_i)+'.h5')
                            hlf = np.array(f.get('HLF'))
                            #select the column with px1, pz1, px2, py2, pz2
                            cols = [5]
                            if i==0:
                                HLF_SIG=hlf[:, cols]
                                i=i+1
                            else:
                                HLF_SIG=np.concatenate((HLF_SIG, hlf[:, cols]), axis=0)
                            f.close()
                            if HLF_SIG.shape[0]>=N_Eft :
                                HLF_SIG=HLF_SIG[:N_Eft, :]
                                break
                        print('HLF_SIG shape')
                        print(HLF_SIG.shape)

                        mass = np.concatenate((HLF_REF, HLF_SIG), axis=0)



                        prediction.append(feature)
                        prediction.append(seed)
                        prediction.append(target)
                        prediction.append(u)
                        prediction.append(v)
                        prediction.append(mass)

                        predictions.append(prediction)
                        prediction = []

                        f.close()

    return predictions







# Structure:
# - feature
# - seed
# - target
# - u
# - v
# - mass
predictions = __read_from_h5_file_predictions__(FILE_PATH, toys)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig=plt.figure(figsize=(20, 16))
plt.clf()

for i in range(len(toys)):
    print(i)
    plt.subplot(4,4,i+1)
    x = predictions[i][5][predictions[i][2]==1][:,0]
    y = predictions[i][0][predictions[i][2]==1][:,0]

    plt.title('NN output $f(M_Z)$ - Toy ' + str(toys[i]), fontsize=fontsize)
    plt.xlabel('$M_{Z}$ [GeV/c$^2$]', fontsize=fontsize)
    plt.ylabel('$f(M_Z)$', fontsize=fontsize)
    plt.hist2d(x,y, bins=100, range=[[0, 800], [-0.25, 5.0]], norm=colors.PowerNorm(0.05))

fig.subplots_adjust(left = 0.03,right = 0.98,bottom = 0.03,top = 0.98,hspace = 0.30)
fig.savefig('./Python/RESULTS/INV_MASS/f_plot_Zeft.pdf')
plt.show()
plt.clf()
fig.clf()
