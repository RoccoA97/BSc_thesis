import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.stats import chi2, chisquare
from scipy.special import erfinv


toys = [5, 9, 19, 20]
verbose = 1

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

for i in range(len(toys)):
    print(i)
    plt.clf()
    fig=plt.figure(figsize=(18, 4.5))
    plt.subplot(1,3,1)
    print(type(predictions[i][5][predictions[i][2]==1]))
    print(predictions[i][5][predictions[i][2]==1].shape)
    x = predictions[i][5][predictions[i][2]==1][:,0]
    y = predictions[i][0][predictions[i][2]==1][:,0]
    print(type(x))
    print(x.shape)
    print(y.shape)
    plt.xlabel('$M_Z$')
    plt.ylabel('$f(M_Z)$')
    plt.hist2d(x,y, bins=100, range=[[0, 800], [-0.25, 2.5]], norm=colors.PowerNorm(0.1))
    #plt.plot(predictions[i][5][predictions[i][2]==1],predictions[i][0][predictions[i][2]==1],'.')
    #plt.xlim(0,350)

    data_mass = predictions[i][5][predictions[i][2]==1]
    data_mass_ref = predictions[i][5][predictions[i][2]==0]

    plt.subplot(1,3,2)
    plt.hist(data_mass, bins=100)#, #range=(250,350))#, density=True)
    plt.yscale('log')


    plt.subplot(1,3,3)
    plt.hist(data_mass_ref, bins=100)#, #range=(250,350))#, density=True)
    plt.yscale('log')

    plt.show()
    plt.clf()
    fig.clf()
