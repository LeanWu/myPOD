import numpy as np
from math import *
import matplotlib.pyplot as plt
import re

#define constants
cAUD = 173.144632674
timeEpoch = 2457597.125
k = 0.01720209895
mu = 1.0
ecA = radians(23.434)
asteroidDes = '1994PN'

#load data

filePath = "test.txt"
data = np.loadtxt(filePath, dtype='bytes', skiprows=1).astype(str)
with open(filePath, 'r') as f:
    asteroidDes = re.search('\((.*)\)', f.readline()).group(1).replace(" ","")

#sexagesimal to decimal
def sexToDecimal(ang, hparam):
    anglist = ang.split(":")
    for i in range(len(anglist)):
        anglist[i] = float(anglist[i])
    decimal = (anglist[1] / 60) + (anglist[2] / 3600)
    if anglist[0] < 0:
        ang = anglist[0] - decimal
    else:
        ang = anglist[0] + decimal
    if hparam == True:
        ang *= 15
    return ang

#decimal to sexagesimal
def decimalToSex(degrees):
    if degrees < 0:
        degrees=-degrees
        hours = int(degrees // 15)
        remaining_degrees = degrees % 15
        minutes = int(remaining_degrees * 4)
        remaining_degrees = (remaining_degrees * 4) % 1
        seconds = float(remaining_degrees * 60)
        return "-"+str(hours).zfill(2)+":"+str(minutes).zfill(2)+":"+f"{seconds:05.2f}"
        # return "-%s:%s:%s" % (hours, minutes, seconds)
    else:
        hours = int(degrees // 15)
        remaining_degrees = degrees % 15
        minutes = int(remaining_degrees * 4)
        remaining_degrees = (remaining_degrees * 4) % 1
        seconds = float(remaining_degrees * 60)
        return str(hours).zfill(2)+":"+str(minutes).zfill(2)+":"+f"{seconds:05.2f}"

#Julian Date Functions
def getJo(date):
    joArr = date.split("-")
    for i in range(len(joArr)):
        joArr[i] = float(joArr[i])
    jodate = 367*joArr[0] - int((7*(joArr[0] + int((joArr[1]+9)/12)))/4) + int(275 * joArr[1] / 9) + joArr[2] + 1721013.5
    return jodate

def getJD(date, time):
    UT = sexToDecimal(time, False)
    return date + (UT / 24)

#parse data
ndata = np.empty([4, 7], dtype='float')
for row in range(data.shape[0]):
    ndata[row,0] = float(getJD(getJo(data[row,0]),data[row,1]))
    ndata[row,2] = radians(sexToDecimal(data[row,2], True))
    ndata[row,3] = radians(sexToDecimal(data[row,3], False))
    ndata[row,4] = float(data[row,4])
    ndata[row,5] = float(data[row,5])
    ndata[row,6] = float(data[row,6])

#define data structure
timeArray = np.copy(ndata[:,0])
raArray = np.copy(ndata[:,2])
decArray = np.copy(ndata[:,3])
R1 = np.array([ndata[0,4],ndata[0,5],ndata[0,6]])
R2 = np.array([ndata[1,4],ndata[1,5],ndata[1,6]])
R3 = np.array([ndata[2,4],ndata[2,5],ndata[2,6]])
R4 = np.array([ndata[3,4],ndata[3,5],ndata[3,6]])

#calculate time intervals
tau = k*(timeArray[2]-timeArray[0])
tau3 = k*(timeArray[2]-timeArray[1])
tau1 = k*(timeArray[0]-timeArray[1])

#calculate rho hats
def hatRhos(rain, decin):
    return np.array([cos(rain)*cos(decin), sin(rain)*cos(decin), sin(decin)])
rhohat1 = hatRhos(raArray[0], decArray[0])
rhohat2 = hatRhos(raArray[1], decArray[1])
rhohat3 = hatRhos(raArray[2], decArray[2])

##START TRUNCATED F+G SERIES FOR INITIAL GUESS (subsitutions galore)
#scalar equations of range pt 1
D0 = rhohat1.dot(np.cross(rhohat2, rhohat3))

D11 = np.cross(R1, rhohat2).dot(rhohat3)
D12 = np.cross(R2, rhohat2).dot(rhohat3)
D13 = np.cross(R3, rhohat2).dot(rhohat3)

D21 = np.cross(rhohat1, R1).dot(rhohat3)
D22 = np.cross(rhohat1, R2).dot(rhohat3)
D23 = np.cross(rhohat1, R3).dot(rhohat3)

D31 = rhohat1.dot(np.cross(rhohat2, R1))
D32 = rhohat1.dot(np.cross(rhohat2, R2))
D33 = rhohat1.dot(np.cross(rhohat2, R3))

A1 = tau3/tau
A3 = -tau1/tau
B1 = A1*(tau**2-tau3**2)/6
B3 = A3*(tau**2-tau1**2)/6
A = -((A1*D21)-D22+(A3*D23))/D0
B = -((B1*D21)+(B3*D23))/D0
E = -2*(rhohat2.dot(R2))
F = np.linalg.norm(R2)**2

#get roots and ask user
roota = -(A**2 + A*E + F)
rootb = -(2*A*B + B*E)
rootc = -(B**2)

coeffs=[rootc,0,0,rootb,0,0,roota,0,1]
roots=np.polynomial.polynomial.polyroots(coeffs)
realRoots = []

for i in range(len(roots)):
    if np.imag(roots[i]) == 0 and np.real(roots[i]) > 0:
        realRoots.append(float(np.real(roots[i])))
print(realRoots)

pick = eval(input("Pick a root to try by index: "))
rmag = realRoots[int(pick)]

u = mu/rmag**3

#get initial f+g
f1 = 1-0.5*u*tau1**2
g1 = tau1-(u*tau1**3)/6
f3 = 1-0.5*u*tau3**2
g3 = tau3-(u*tau3**3)/6

C2 = -1
C1 = g3/(f1*g3-f3*g1)
C3 = -g1/(f1*g3-f3*g1)

rho1 = abs((C1*D11+C2*D12+C3*D13)/(C1*D0))
rho2 = abs((C1*D21+C2*D22+C3*D23)/(C2*D0))
rho3 = abs((C1*D31+C2*D32+C3*D33)/(C3*D0))

#get initial position vectors
r1 = rhohat1 * rho1 - R1
r2 = rhohat2 * rho2 - R2
r3 = rhohat3 * rho3 - R3

#get initial velocity vector
d1 = -f3/(f1*g3-f3*g1)
d3 = f1/(f1*g3-f3*g1)
r2dot = d1*r1 + d3*r3

#iteration control variables
tolerance = 1e-12
rOld = 0
counter = 0
tau1c = tau1
tau3c = tau3
tauc = tau

#MAIN ITERATION
print("""
--------------------------------------------------------------------------
Beginning Iteration
Tolerance: %s
--------------------------------------------------------------------------""" % (tolerance))
while abs(rOld-rmag) > tolerance:
    counter += 1

    #closed form f+g functions
    a = ((2/rmag)-np.dot(r2dot,r2dot))**-1
    n = 1/sqrt(a**3)
    sub1 = 1-rmag/a
    sub2 = rmag*np.linalg.norm(r2dot)/n/a/a

    #newtons method to find delta E
    x1 = n*tau1c
    x1old = 0
    while abs(x1-x1old)>0.0000000001:
        x1old = x1
        fx1 = x1old-sub1*sin(x1old)+sub2*(1-cos(x1old))-n*tau1c
        fprimex1 = 1-sub1*cos(x1old)+sub2*sin(x1old)
        x1 = x1old-fx1/fprimex1
    x3 = n*tau3c
    x3old = 0
    while abs(x3-x3old)>0.0000000001:
        x3old = x3
        fx3 = x3old-sub1*sin(x3old)+sub2*(1-cos(x3old))-n*tau3c
        fprimex3 = 1-sub1*cos(x3old)+sub2*sin(x3old)
        x3 = x3old-fx3/fprimex3

    #calculte new f+g
    f1 = 1-a*(1-cos(x1))/rmag
    f3 = 1-a*(1-cos(x3))/rmag
    g1 = tau1c+(sin(x1)-x1)/n
    g3 = tau3c+(sin(x3)-x3)/n

    C1 = g3/(f1*g3-f3*g1)
    C3 = -g1/(f1*g3-f3*g1)
    d1 = -f3/(f1*g3-f3*g1)
    d3 = f1/(f1*g3-f3*g1)

    rho1 = abs((C1*D11+C2*D12+C3*D13)/(C1*D0))
    rho2 = abs((C1*D21+C2*D22+C3*D23)/(C2*D0))
    rho3 = abs((C1*D31+C2*D32+C3*D33)/(C3*D0))

    #get new state vectors
    r1 = rhohat1 * rho1 - R1
    r2 = rhohat2 * rho2 - R2
    r3 = rhohat3 * rho3 - R3
    r2dot = d1*r1 + d3*r3

    rOld = rmag + 0
    rmag = np.linalg.norm(r2)

    #debug
    print("Iteration %s: delta r = %s" % (counter, abs(rmag-rOld)))

    #light time correction
    tauc = k*((timeArray[2]-rho3/cAUD)-(timeArray[0]-rho1/cAUD))
    tau3c = k*((timeArray[2]-rho3/cAUD)-(timeArray[1]-rho2/cAUD))
    tau1c = k*((timeArray[0]-rho1/cAUD)-(timeArray[1]-rho2/cAUD))

##output
r2dot = k*r2dot