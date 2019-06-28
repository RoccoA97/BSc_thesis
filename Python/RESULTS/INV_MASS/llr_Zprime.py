import sys
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.stats import chi2, chisquare
from scipy.special import erfinv


toys = [5, 9, 16, 18,
        19, 20, 21, 23,
        26, 40, 43, 44,
        45, 47, 48, 51]
verbose = 1
fontsize = 13

print(len(toys))
FILE_PATH = 'C:\\Users\\Ardino Pierfrancesco\\Desktop\\Z_5D_SIMULATIONS_RESULTS-master\\wcliptest\\ref1000000_bkg20000_sig40\\epochs300000\\Z_5D_patience5000_ref1000000_bkg20000_sig40_epochs300000_latent5_wclip2.7_run1\\'


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
    print(type(predictions[i][5][predictions[i][2]==1]))
    print(predictions[i][5][predictions[i][2]==1].shape)
    x = predictions[i][5][predictions[i][2]==1][:,0]
    y = predictions[i][0][predictions[i][2]==1][:,0]
    print(type(x))
    print(x.shape)
    print(y.shape)

    plt.title('NN output $f(M_Z)$ - Toy ' + str(toys[i]), fontsize=fontsize)
    plt.xlabel('$M_Z$', fontsize=fontsize)
    plt.ylabel('$f(M_Z)$', fontsize=fontsize)
    plt.hist2d(x,y, bins=100, range=[[0, 400], [-0.25, 3.0]], norm=colors.PowerNorm(0.05))
    #plt.plot(predictions[i][5][predictions[i][2]==1],predictions[i][0][predictions[i][2]==1],'.')
    #plt.xlim(0,350)

    """
    data_mass = predictions[i][5][predictions[i][2]==1]
    data_mass_ref = predictions[i][5][predictions[i][2]==0]

    plt.subplot(1,3,2)
    plt.hist(data_mass, bins=100)#, #range=(250,350))#, density=True)
    plt.yscale('log')


    plt.subplot(1,3,3)
    plt.hist(data_mass_ref, bins=100)#, #range=(250,350))#, density=True)
    plt.yscale('log')
    """

fig.subplots_adjust(left = 0.03,right = 0.98,bottom = 0.03,top = 0.98,hspace = 0.30)
fig.savefig('./Python/RESULTS/INV_MASS/f_plot_Zprime.pdf')
plt.show()
plt.clf()
fig.clf()
