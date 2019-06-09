import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from math import sqrt, atan

def B(omega, omega_0, gamma, Gamma=1):
    return Gamma / np.sqrt( (omega_0**2 - omega**2)**2 + 4*(gamma**2)*(omega_0**2)*(omega**2))

def delta(omega, omega_0, gamma, Gamma=1):
    return np.arctan((2*gamma*omega_0*omega)/(omega_0**2 - omega**2))

omega_0 = 1.0
gamma = [0.1,0.15,0.2,0.25,0.3]
Gamma = 1.0

x1 = np.arange(0.0, 2.0, 0.01)
x2 = np.arange(-np.pi/2, np.pi/2, 0.01)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure(1, figsize=(12, 5))
plt.subplot(121)
plt.title('Amplitude B')
plt.xlabel('$\Omega$')
plt.ylabel('B')
for i in range(len(gamma)):
    label = '$\gamma$ = ' + str(gamma[i])
    print(label)
    plt.plot(x1, B(x1,omega_0, gamma[i], Gamma),label=label)
plt.legend()

plt.subplot(122)
plt.title('Phase shift $\delta$')
plt.xlabel('$\Omega$')
plt.ylabel('$\delta$')
for i in range(len(gamma)):
    label = '$\gamma$ = ' + str(gamma[i])
    plt.plot(x1, delta(x1,omega_0, gamma[i], Gamma),label=label)
plt.legend()
plt.show()

os.getcwd()

fig.savefig('./Python/LHC/resonance.pdf')
