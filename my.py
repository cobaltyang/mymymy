import numpy as np
from scipy.linalg import eigh, svd
from scipy.fft import fft
import cvxpy as cp
import pdb
# import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
import math
from numpy.random import rand
import pprint
from scipy import io

def tpuconnect():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_strategy = tf.distribute.TPUStrategy(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    return tpu_strategy
# 参数定义


f0 = 8e9  # 信号中心频率8GHz
fr = 8e9  # 干扰信号频率8GHz
B = 5e8  # 信号带宽500MHz
fl = f0 - B / 2  # 信号起始频率
fh = f0 + B / 2  # 信号最高频率
fs = 3 * f0  # 采样频率
# -------------------------固定常数
pi = math.pi
radians = pi / 180
M = 16  # 阵元数为M
P = 3  # 信号数目
c = 3e8  # 光速
d = 0.5 * c / f0  # 阵间距

# ---------------------信号时间和信号点数
T = 5e-07
Nr = round(T * fs)
J = 1200
# ---------------------------------------------------角度和信噪比
theta1 = np.arange(-15, 16, 1)  # 期望信号区域
theta2 = np.arange(-60, -29, 1)  # 宽带干扰区域
theta3 = np.arange(30, 61, 1)  # 窄带干扰区域
snr = np.array([10, 30, 30])  # 信噪比
sensor_error = 0.1 * (rand() - 0.5)  # 阵元位置误差


def calculate_frequency_parameters():

    t = np.arange(0, Nr) / fs  # 时间轴刻度
    fw = np.linspace(0, fs, Nr)  # 频率轴刻度
    m = int(f0 / fs)
    kn = np.nonzero(np.logical_and((fw >= f0 - m * fs - B / 2) , (fw <= f0 - m * fs + B / 2))
                 )[0] # fft后落在带宽内频率索引
    G = len(kn)  # 落在带宽内的频率的个数
    F = fw[kn]  # 落在带宽内的频率

    return G, F, kn


def generate_DOA_combinations():
    k = 0
    DOA_train = np.zeros((3,1,len(theta1)*len(theta1)*len(theta1)))
    for i1 in range(len(theta1)):
        for i2 in range(len(theta2)):
            for i3 in range(len(theta3)):
                auv = np.array([theta1[i1], theta2[i2], theta3[i3]])
                auv = auv[:,np.newaxis]
                DOA_train[:,:,k] = auv
                k += 1
    i = 0
    new_DOA = []
    for uu in range(len(theta1) * len(theta2) * len(theta3)):
        theta = DOA_train[:,:,uu]
        if abs(theta[0] - theta[1]) <= 15 or abs(theta[0] - theta[2]) <= 15:
            continue
        else:
            new_DOA.append(theta)
            i += 1
    new_DOA = np.array(new_DOA)
    new_DOA = np.transpose(new_DOA, axes=(1, 2, 0)) 
    return new_DOA


G, F, kn = calculate_frequency_parameters()
new_DOA = generate_DOA_combinations()
# 1.产生信号


def LFM_source(theta, snr):  #两个宽带信号频率是一样的
    t = np.arange(0, T, 1/fs)       # 时间变量
    P = 10**(snr/20)                # 信号功率
    K = B / T                       # 调频速率
    x = np.zeros((M,len(t)))
    for vv in range(M):
        theta_rad = theta*radians
        yanqian = 2*pi*fl*(t-(vv)*d*np.sin(theta_rad)/c)
        yanhou = pi*K*(t-(vv)*d*np.sin(theta_rad)/c)**2
        x[vv,:] = P*np.exp(1j*(yanqian+yanhou))
    return x


def arrayline(thetacom, fpin,sensor_error=0):  
    return np.exp(-1j * 2 * pi * (d + sensor_error) * fpin * np.sin(thetacom * radians) * np.arange(M) / c)


def zhaidai(sensor_error, thetacom):
    t = np.arange(Nr) / fs  # 窄带干扰专用
    s = np.sqrt(10 ** (snr[2] / 10)) * np.sin(2 * pi * f0 * t)
    s = s[np.newaxis,:]
    a = arrayline(thetacom, f0,sensor_error).conj()
    a = a[:, np.newaxis]

    signal = a @ s  # a*s
    return signal


def generate_signal(thetacom, sensor_error):
    x = LFM_source(thetacom[0], snr[0])  # 期望信号
    x += LFM_source(thetacom[1], snr[1])  # 宽带干扰
    u = zhaidai(sensor_error, thetacom[2])
    x += u# 窄带干扰
    noise = 1 / np.sqrt(2) * np.random.randn(M, Nr) + 1j / \
        np.sqrt(2) * np.random.randn(M, Nr)  # 加噪声
    x += noise
    return x
# 2.计算fft


def calculate_fft(x):
    K_freq = Nr//J

    X = np.zeros((M, K_freq, J), dtype=complex)
    for m in range(M):
        for k in range(K_freq):
            X[m, k, :] = np.fft.fft(x[m, (k*J):((k+1)*J)], J)  # 1200点，所以是1200

    return X
# 3.估计子带协方差矩阵


def xiefangcha(X, mm, kn):
    K_freq = Nr//J
    Rfl = np.zeros((M, M, G), dtype=complex)
    for g in range(G):
        Rfl[:,:, g] = X[:,kn[g]%K_freq,kn[g]//K_freq] @ X[:,kn[g]%K_freq,kn[g]//K_freq].conj().T

    return Rfl
# 4.计算聚焦矩阵


def calculate_exponential(f, m, theta):
    return np.exp(-1j * 2 * pi * d * f * m * np.sin(theta*radians) / c)  #这里只返回一个数


def calculate_Y():
    step = 1  # 空间谱步长
    theta = np.arange(-90, 91, step)
    theta_len = len(theta)

    Af0 = np.zeros((M, theta_len), dtype=complex)
    for theta_index in range(theta_len):
        for m in range(M):
            Af0[m, theta_index] = calculate_exponential(
                f0, m, theta[theta_index]).T

    Y = np.zeros((M, M, G), dtype=complex)
    for g in range(G):  # G个频点
        Af = np.zeros((M, theta_len), dtype=complex)
        for theta_index in range(theta_len):
            for m in range(M):
                Af[m, theta_index] = calculate_exponential(
                    F[g], m, theta[theta_index]).T

        U, _, V = np.linalg.svd(Af.dot(Af0.conj().T))
        V = V.T
        Y[:, :, g] = V.dot(U.conj().T)

    return Y
# 5&9.聚焦&叠加


def JuDie(Y, Ju):
    Rin = np.zeros((M, M), dtype=complex)
    for g in range(G):
        Rin += Y[:, :, g] @ Ju[:, :, g] @ Y[:, :, g].conj().T
    Rin = Rin/G
    return Rin
# 6.重构子带协方差矩阵R


def ChongGou(thetacom, R_):
    doa_i = thetacom[1:].T

    _, D = np.linalg.eig(R_)
    D = np.diag(D)
    lambda_max = np.max(D)
    lambda_min = np.min(D)

    Rinfl_U = np.zeros((M, M, G), dtype=complex)

    for g in range(G):
        Rinfl_U[:, :, g] = lambda_max * (arrayline(doa_i[0],F[g])*arrayline(doa_i[0],F[g]).conj().T) + lambda_max * (arrayline(doa_i[1],F[g])*arrayline(doa_i[1],F[g]).conj().T)+lambda_min * np.eye(M)

    return Rinfl_U
# 7.锥化来零陷展宽


def taperize(RflG):
    Ufl = np.zeros((M, M, G))
    delta = 0.5 * radians

    for g in range(G):
        for m1 in range(M):
            for m2 in range(M):
                Ufl[m1, m2, g] = 1 + 2 * np.cos(2 * pi * F[g] * d * delta * (m1 - m2) / c)

    return Ufl*RflG
# 11.对导向矢量求优化问题修正


def solve_optimization_problem(a_except, Rin):

    e = cp.Variable((M, 1), complex=True)  # Define the optimization variables

    # Define the objective function and constraints
    obj = cp.Minimize(cp.quad_form((a_except + e), np.linalg.inv(Rin)))
    constr = [cp.quad_form((a_except + e), Rin) <= cp.quad_form(a_except, Rin),
              cp.norm(a_except + e) <= np.sqrt(M),
              a_except.conj().T @ e == 0]

    prob = cp.Problem(obj, constr)  # Solve the optimization problem
    prob.solve()

    if prob.status == cp.OPTIMAL:   # Check if the problem was successfully solved

        a_except = a_except + e.value  # Update the value of a_except
    else:
        print("Problem not solved.")

    return a_except
 # 重点！！


def datasetgenerate(new_DOA):
    for mm in [0, 1]:  # range(len(new_DOA)): # 对于每个角度组合
        thetacom = new_DOA[:,:,mm]  # 每一个角度组合
        x = generate_signal(thetacom, sensor_error)  # 1.生成信号
        X = calculate_fft(x)  # 2.傅里叶变换
        Rfl = xiefangcha(X, mm, kn)  # 3.计算子带协方差矩阵Rfl

        Y = calculate_Y()  # 4.计算聚焦矩阵Y
        R_ = JuDie(Y, Rfl)  # 5.Rfl聚焦&叠加，利用R_来重构

        RflG = ChongGou(thetacom, R_)  # 6.重构子带协方差矩阵Rinfl_U

        RflK = taperize(RflG)  # 7.乘以锥化矩阵进行零陷展宽

        Rin = JuDie(Y, RflK)  # 9.把Rinfl_U聚焦&叠加
    # 二、导向矢量修正
        a_except = arrayline(thetacom[0], f0)
        a_except = a_except[:, np.newaxis]
        a_except = solve_optimization_problem(a_except, Rin)  # 11.对导向矢量求优化问题修正

        Wcsm = np.linalg.inv(Rin) @ a_except / (a_except.T @
                                            np.linalg.inv(Rin) @ a_except)  # 三、求权重
        w_out[:, mm] = Wcsm
        return Rin, w_out


def process_data(new_DOA, R_in, w_out):
    j_test = 0
    j_train = 0
    doa_test = []
    doa_train = []
    R_test_real = []
    R_test_imag = []
    R_train_real = []
    R_train_imag = []
    w_test_real = []
    w_test_imag = []
    w_train_real = []
    w_train_imag = []

    for i in range(new_DOA.shape[2]):
        if (i + 1) % 10 == 0:
            doa_test.append(new_DOA[:, :, i])

            # 归一化
            R_real = np.real(R_in[:, :, i])
            a = np.max(R_real)
            b = np.min(R_real)
            R_r_nor = (R_real - b) / (a - b)

            R_imag = np.imag(R_in[:, :, i])
            a = np.max(R_imag)
            b = np.min(R_imag)
            R_i_nor = (R_imag - b) / (a - b)

            R_test_real.append(R_r_nor)
            R_test_imag.append(R_i_nor)
            w_test_real.append(np.real(w_out[:, :, i]))
            w_test_imag.append(np.imag(w_out[:, :, i]))

            j_test += 1
        else:
            doa_train.append(new_DOA[:, :, i])

            # 归一化
            R_real = np.real(R_in[:, :, i])
            a = np.max(R_real)
            b = np.min(R_real)
            R_r_nor = (R_real - b) / (a - b)

            R_imag = np.imag(R_in[:, :, i])
            a = np.max(R_imag)
            b = np.min(R_imag)
            R_i_nor = (R_imag - b) / (a - b)

            R_train_real.append(R_r_nor)
            R_train_imag.append(R_i_nor)
            w_train_real.append(np.real(w_out[:, :, i]))
            w_train_imag.append(np.imag(w_out[:, :, i]))

            j_train += 1

    R_train_real = np.array(R_train_real)
    R_train_imag = np.array(R_train_imag)
    w_train_real = np.array(w_train_real)
    w_train_imag = np.array(w_train_imag)
    doa_train = np.array(doa_train)

    R_test_real = np.array(R_test_real)
    R_test_imag = np.array(R_test_imag)
    w_test_real = np.array(w_test_real)
    w_test_imag = np.array(w_test_imag)
    doa_test = np.array(doa_test)
    return R_train_real, R_train_imag, w_train_real, w_train_imag, R_test_real, R_test_imag, w_test_real, w_test_imag

# @tf.function


def main():
    Rin, w_out = datasetgenerate(new_DOA)

    # R_train_real,R_train_imag,w_train_real,w_train_imag,R_test_real,R_test_imag,w_test_real,w_test_imag = process_data(new_DOA, R_in, w_out)
# strategy = tpuconnect()
# with strategy.scope():
main()
