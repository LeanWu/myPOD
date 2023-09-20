from scipy.optimize import least_squares
import poliastro.constants.general as constant
import numpy as np
import dynamics

# 计算预测值与观测值之间的误差
def err_OC(x_observed,rv,t,index):
    err=np.zeros(len(t))
    for i in range(len(t)):
        rv_computed=dynamics.mypropagation(rv,t[i]-t[index],constant.GM_earth.value,-1)
        err_vector=x_observed[i,:]-rv_computed[1,0:3]
        err[i]=np.linalg.norm(err_vector)
    return err

# 最小二乘法求解
def optimize_orb(x_observed,rv0,t,index):
    new_err_OC=lambda rv:err_OC(x_observed,rv,t,index)
    result = least_squares(
        new_err_OC, rv0, method='dogbox', xtol=1e-9)
    # return rv0
    return result.x
