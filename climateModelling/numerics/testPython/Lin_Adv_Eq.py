###################
import numpy as np
import matplotlib.pyplot as plt


#Equation definition

def initialBell(x):
              return np.where(x%1. <0.5, np.power(np.sin(2*x*np.pi), 2), 0)


#setup space, initial phi prfile and Courant number
nx = 40   #points in space
c = 1   #courant number

#spatial variable going from zero to one inclusive
x = np.linspace(0.0, 1.0, nx+1)

#three time levels of the dependent variable, phi
phi = initialBell(x)
phiNew = phi.copy()
phiOld = phi.copy()

#FCTS for the first time-step 
# loop over space

for j in xrange(1,nx):
         phi[j] = phiOld[j] - 0.5*c*(phiOld[j+1] - phiOld[j-1])

#apply periodic bundary conditions
phi[0] = phiOld[0] - 0.5*c*(phiOld[1] - phiOld[nx-1])
phi[nx] = phi[0]

#Loop over remaining time-steps (nt) using CTCS
nt = 40
for n in xrange(1,nt):
      #loop over space
      for j in xrange(1,nx):
               phiNew[j] = phiOld[j] - c*(phi[j+1] - phi[j-1])
      #apply periodic boundary conditions
      phiNew[0] = phiOld[0] - c*(phi[1] - phi[nx-1])
      phiNew[nx] = phiNew[0]

#update phi for the next time-step
phiOld = phi.copy()
phi = phiNew.copy()

#derived quantities
u = 1.
dx = 1./nx
dt = c*dx/u
t = nt*dt

#Plot
plt.plot(x, initialBell(x - u*t), 'k', label='analytic')
plt.plot(x, phi, 'b', label='CTCS')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('$\phi$')
plt.axhline(0, linestyle=':', color='black')
plt.show()










