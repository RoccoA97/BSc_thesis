import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from math import sqrt, atan


fontsize=15



def B(omega, omega_0, Gamma, A=1):
    return A / ( (omega**2 - omega_0**2)**2 + Gamma**2)

omega_0 = 1.0
Gamma = [0.10,0.125,0.15,0.175,0.20,0.25,0.30]
A = 1.0

x1 = np.arange(0.0, 2.0, 0.001)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure(1, figsize=(8, 6))
plt.title('Breit-Wigner', fontsize=fontsize)
plt.xlabel('$\mathcal{E}$', fontsize=fontsize)
plt.ylabel('$f(\mathcal{E})$', fontsize=fontsize)
plt.xlim(0.7,1.3)
for i in range(len(Gamma)):
    label = '$\Gamma$ = ' + str(Gamma[i])
    print(label)
    plt.plot(x1, B(x1,omega_0, Gamma[i], A),linewidth=1.5,label=label)
plt.legend(fontsize=fontsize)
plt.show()

os.getcwd()

fig.subplots_adjust(left = 0.08,right = 0.92,bottom = 0.08,top = 0.95)
fig.savefig('./Python/LHC/bw.pdf')
