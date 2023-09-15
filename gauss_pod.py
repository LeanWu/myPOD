import numpy as np
from math import *
import poliastro.constants.general as constant

# gauss法初轨确定
def gauss_preOD(t,rho_direction,station_position,mu):
    #define constants
    #calculate time intervals
    tau = t[2]-t[0]
    tau3 = t[2]-t[1]
    tau1 = t[0]-t[1]

    # input
    rhohat1=rho_direction[0,:]
    rhohat2=rho_direction[1,:]
    rhohat3=rho_direction[2,:]
    R1=station_position
    R2=station_position
    R3=station_position

    ##START TRUNCATED F+G SERIES FOR INITIAL GUESS (subsitutions galore)
    #scalar equations of range pt 1
    p1=np.cross(rhohat2,rhohat3)
    p2=np.cross(rhohat1,rhohat3)
    p3=np.cross(rhohat1,rhohat2)

    D0 = np.dot(rhohat1,p1)

    D11 = np.dot(R1,p1)
    D12 = np.dot(R1,p2)
    D13 = np.dot(R1,p3)

    D21 = np.dot(R2,p1)
    D22 = np.dot(R2,p2)
    D23 = np.dot(R2,p3)

    D31 = np.dot(R3,p1)
    D32 = np.dot(R3,p2)
    D33 = np.dot(R3,p3)

    A1 = tau3/tau
    A3 = tau1/tau
    B1 = A1*(tau**2-tau3**2)/6
    B3 = A3*(tau**2-tau1**2)/6
    A = (-(A1*D12)+D22+(A3*D32))/D0
    B = (-(B1*D12)+(B3*D32))/D0
    E = rhohat2.dot(R2)
    F = np.linalg.norm(R2)**2

    #get roots and ask user
    roota = -(A**2 + 2*A*E + F)
    rootb = -2*mu*B*(A+E)
    rootc = -((mu*B)**2)

    coeffs=[rootc,0,0,rootb,0,0,roota,0,1]
    roots=np.polynomial.polynomial.polyroots(coeffs)
    realRoots = []

    for i in range(len(roots)):
        if np.imag(roots[i]) == 0 and np.real(roots[i]) > 0:
            realRoots.append(float(np.real(roots[i])))
    print(realRoots)

    pick = eval(input("Pick a root to try by index: "))
    rmag = realRoots[int(pick)]  

    #get initial rho
    u = mu/rmag**3
    f1 = 1-0.5*u*tau1**2
    g1 = tau1-(u*tau1**3)/6
    f3 = 1-0.5*u*tau3**2
    g3 = tau3-(u*tau3**3)/6

    rho2 = A + mu*B/rmag**3
    rho1 = 1/D0*((6*(D31*tau1/tau3 + D21*tau/tau3)*rmag**3
        + mu*D31*(tau**2 - tau1**2)*tau1/tau3)
        /(6*rmag**3 + mu*(tau**2 - tau3**2)) - D11)
    rho3 = 1/D0*((6*(D13*tau3/tau1 - D23*tau/tau1)*rmag**3
        + mu*D13*(tau**2 - tau3**2)*tau3/tau1)
        /(6*rmag**3 + mu*(tau**2 - tau3**2)) - D33)

    #get initial position vectors
    r1 = rhohat1 * rho1 + R1
    r2 = rhohat2 * rho2 + R2
    r3 = rhohat3 * rho3 + R3

    #get initial velocity vector
    r2dot = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)

    #output
    # print(r2)
    # print(r2dot)
    return r2,r2dot
