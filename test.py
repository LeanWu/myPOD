import dynamics
import least_squares_od
import gauss_pod
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from astropy import units as u
from astropy.time import Time

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import poliastro.constants.general as constant

# 基本参数
mu=constant.GM_earth
Re=constant.R_earth.to(u.m)

# 卫星初始状态
a = Re+20000 * u.km
ecc = 0.0 * u.one
inc = 30 * u.deg
raan = 0 * u.deg
argp = 0 * u.deg
nu = 0 * u.deg
time=Time('2020-01-01 00:00:00',format='iso', scale='utc')

# 卫星递推
orb=Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, time)
r0=orb.r.to(u.m)
v0=orb.v.to(u.m/u.second)
rv0=np.array([r0.value[0],r0.value[1],r0.value[2],v0.value[0],v0.value[1],v0.value[2]])
print('Orbit Period:',orb.period.to(u.hour))

t_max=3600
t_interval=1800
t=np.linspace(0,t_max,t_interval)
rv=dynamics.mypropagation(rv0,t_max,mu.value,t_interval)
# print(rv)

# 测站位置
longitude=math.radians(110)
latitude=math.radians(35)
station_position=np.array([math.cos(latitude)*math.cos(longitude), math.cos(latitude)*math.sin(longitude), math.sin(latitude)])*Re.value
# print(station_position)

rho_direction=np.zeros((3,3))
for i in range(3):
    rho=rv[i,0:3]-station_position
    rho_direction[i,:]=rho/np.linalg.norm(rho)
# print(rho_direction)

# gauss初轨确定
t=np.array([0,1800,3600])
r2,r2dot=gauss_pod.gauss_preOD(t,rho_direction,station_position,mu.value)
# print(rv)

# 初轨结果展示
rv2=np.array([r2[0],r2[1],r2[2],r2dot[0],r2dot[1],r2dot[2]])
rv_c=dynamics.mypropagation(rv2,orb.period.to(u.second),mu.value,orb.period.to(u.second)/100)

x=np.zeros(100)
y=np.zeros(100)
z=np.zeros(100)
for i in range(100):
    x[i]=rv_c[i,0]
    y[i]=rv_c[i,1]
    z[i]=rv_c[i,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.scatter(rv[0,0], rv[0,1], rv[0,2], c='red', s=5, label='Highlighted Point')
ax.scatter(rv[1,0], rv[1,1], rv[1,2], c='red', s=5, label='Highlighted Point')
ax.scatter(rv[2,0], rv[2,1], rv[2,2], c='red', s=5, label='Highlighted Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 最小二乘法改进
rv2_optimized=least_squares_od.optimize_orb(rv[:,0:3],rv2,t,1)
err_r=np.linalg.norm(rv2_optimized[0:3]-rv[1,0:3])
err_v=np.linalg.norm(rv2_optimized[3:6]-rv[1,3:6])
print("position err:",err_r)
print("velocity err:",err_v)

# 改进轨道结果展示
T=orb.period.to(u.second).value+3600
rv_c=dynamics.mypropagation(rv2_optimized,T,mu.value,T/100)

x=np.zeros(100)
y=np.zeros(100)
z=np.zeros(100)
for i in range(100):
    x[i]=rv_c[i,0]
    y[i]=rv_c[i,1]
    z[i]=rv_c[i,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.scatter(rv[0,0], rv[0,1], rv[0,2], c='red', s=5, label='Highlighted Point')
ax.scatter(rv[1,0], rv[1,1], rv[1,2], c='red', s=5, label='Highlighted Point')
ax.scatter(rv[2,0], rv[2,1], rv[2,2], c='red', s=5, label='Highlighted Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()