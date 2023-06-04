import numpy as np 
import matplotlib.pyplot as plt

# 参数设置
f0 = 8*10**9   # 信号中心频率8GHz
fr = 8*10**9     # 干扰信号频率8GHz
B = 500*10**6      # 信号带宽500MHz
fl = f0-B/2          # 信号起始频率
fh = f0+B/2         # 信号最高频率
Tr = 50*10**(-8)   # 工作周期
fs = 3*f0             # 采样频率
snr = [10,30,30]     # 信噪比
M = 16                    # 阵元数为M
P = 3               # 信号数目
c = 3*10**8           # 光速
d = 0.5*c/f0        # 阵间距
Nr = int(Tr*fs)     # 一个工作周期内的信号点数
t = np.linspace(0, Nr-1, Nr)/fs     # 时间轴刻度
fw = np.linspace(0,fs,Nr)            # 频率轴刻度
m = np.fix(f0/fs)
kn = np.where((fw >= f0-m*fs-B/2) & (fw <= f0-m*fs+B/2))
G = len(kn)           # 落在带宽内的频率的个数 
F = fw[kn]          # 落在带宽内的频率
J = 1200                  # FFT的点数
K_freq = Nr/J        # 频域快拍

# 加载角度集
doa = np.load('DOA_Set.npy')  

for mm in range(doa.shape[2]):

    data = doa[:,:,mm]   # 每一个角度组合
    edata = data+ 1*(2*np.random.rand(3)-1)   # 角度误差  
    
    # 产生阵列接收的宽带数据            
    # 叠加期望信号
    x = LFM_source(M,fh,fl,B,fs,Tr,data[0],snr[0]) 
    # 叠加宽带干扰信号
    x = x + LFM_source(M,fh,fl,B,fs,Tr,data[1],snr[1])
    # 叠加窄带干扰
    x = x + np.exp(-1j*2*np.pi*d*f0*np.sin(data[2]*(0:M-1)/c)) * np.sqrt(10**(snr[2]/10))*np.sin(2*np.pi*f0*t)
    # 叠加噪声
    noise = 1/np.sqrt(2)*np.random.randn(M,Nr)+1/np.sqrt(2)*1j*np.random.randn(M,Nr)  # 噪声
    x = x + noise
    
    # 傅里叶变换
    X = np.zeros((M,K_freq,J), dtype=np.complex_)
    for m in range(M):
        for k in range(K_freq):
            X[m,k,:] = np.fft.fft(x[m,((k-1)*J+1):k*J],J)
            
    # 估计子带协方差矩阵
    Rfl = np.zeros((M,M,G), dtype=np.complex_)
    for g in range(G):
        Rfl[:,:,g] = X[:,kn[g]]*np.conj(X[:,kn[g]]).T  

    # 聚焦变换,计算子带重构所用最大最小特征值
    # 计算聚焦矩阵
    step = 1  # 空间谱步长
    theta = np.arange(-90, 90+step, step)
    theta_len = len(theta)
    Af0 = np.zeros((M,theta_len), dtype=np.complex_)
    for theta_index in range(theta_len):
        for m in range(M):
            Af0[m,theta_index] = np.exp(-1j*2*np.pi*d*f0*m*np.sin(theta[theta_index]*np.pi/180)/c)
            
    Y = np.zeros((M,M,G), dtype=np.complex_) 
    for g in range(G):  
        # 其他频点的方向向量矩阵
        Af = np.zeros((M,theta_len), dtype=np.complex_)
        for theta_index in range(theta_len):
            for m in range(M):
                Af[m,theta_index] = np.exp(-1j*2*np.pi*d*F[g]*m*np.sin(theta[theta_index]*np.pi/180)/c)
        # 计算聚焦矩阵
        U, _, V = np.linalg.svd(Af @  np.conj(Af0).T)
        Y[:,:,g] = V @ U.conj().T 
        
    # 聚焦&叠加
    R_ = np.zeros((M,M,G), dtype=np.complex_)
    for g in range(G):
        R_[:,:,g]= Y[:,:,g] @ Rfl[:,:,g] @ np.conj(Y[:,:,g]).T  
    R_ = np.sum(R_, axis=2)/G  

    # 重构子带协方差矩阵r
    doa_i = data[1:]
    D, _ = np.linalg.eig(R_)
    D = np.diag(D)
    lambda_max = np.max(D) 
    lambda_min = np.min(D) 
    Rinfl_U = np.zeros((M,M,G), dtype=np.complex_)
    for g in range(G):
        Rinfl_U[:,:,g] = lambda_max*(np.exp(-1j*2*np.pi*d*F[g]*(np.arange(M)-M//2)*np.sin(doa_i*np.pi/180)/c) 
                                      @ np.exp(-1j*2*np.pi*d*F[g]*(np.arange(M)-M//2)*np.sin(doa_i*np.pi/180)/c).T) \
                             + lambda_min*np.eye(M) 
        
    # 零陷展宽
    # 计算锥化矩阵
    Ufl = np.zeros((M,M,G), dtype=np.complex_)
    delta = 0.5*np.pi/180
    for g in range(G):
        for m1 in range(M):
            for m2 in range(M):
                Ufl[m1,m2,g] = 1 + 2*np.cos(2*np.pi*F[g]*d*delta*(m1-m2)/c) 
                 
    # 零陷展宽
    Rinfl_U = Rinfl_U * Ufl  
        
    # 聚焦&叠加
    Rin = np.zeros((M,M), dtype=np.complex_) 
    for g in range(G):
        Rin = Rin + Y[:,:,g] @ (Rinfl_U[:,:,g] @ np.conj(Y[:,:,g]).T)  
    Rin = Rin/G  
    
# 波束形成
    a_except = np.exp(-1j*np.pi*(np.arange(M)-M//2)*np.sin(data[0]*np.pi/180))

    from scipy.optimize import minimize

    def cost_func(e):
        return np.real(quad_form(a_except+e, np.linalg.inv(Rin)))

    cons = ({'type': 'ineq', 'fun': lambda e: quad_form(a_except+e, Rin) - quad_form(a_except, Rin)},  
        {'type': 'ineq', 'fun': lambda e: np.linalg.norm(a_except + e) - np.sqrt(M)},
        {'type': 'eq', 'fun': lambda e: a_except.conj().T @ e})

    res = minimize(cost_func, np.zeros(M, dtype=np.complex_), constraints=cons)
    e = res.x

    a_except = a_except + e
    
    Wcsm = np.linalg.pinv(Rin) @ a_except / (a_except.conj().T @ np.linalg.pinv(Rin) @ a_except)
    
    # 计算波束图
    f_csm = np.zeros((G,theta_len))
    for g in range(G):
        for theta_index in range(theta_len):
            f_csm[g,theta_index] = Wcsm.conj().T @ np.exp(-1j*2*np.pi*d*F[g]/c*(np.arange(M)-M//2)*np.sin(theta[theta_index]*np.pi/180)) 
            f_csm[g,theta_index