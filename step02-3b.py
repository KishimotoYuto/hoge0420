import numpy as np
import random as rd
import MGD2
import matplotlib.pyplot as plt

# クラス
K = 2

# 次元
D = 6

# x:D次元のデータ　lab:ラベル　t:教師信号　の取得  
x, lab, t = MGD2.getData(K, D)

# データ数
N = len(x)

# NN * N 試行回数
NN = 1000

# パラメータ(D+1個)
theta = np.zeros(D+1)
for i in range(len(theta)):
    theta[i] = rd.random() * 0.02 -0.01

# eta
eta = 0.001

# 誤識別率
result = np.zeros(NN)

# 交差エントロピーの平均
H = np.zeros(NN)

# シグモイド関数
def fun_z(s):
    return 1.0 / (1+np.exp(-1*s))

# 規準
def fun_h(t, z):
    return -1*t*np.log(z) - (1-t)*np.log(1-z)

# ロジスティック回帰による識別
for i in range(N*NN+1):
    # N個のデータの中からランダムに選択
    num = rd.randint(0, N-1)

    y = np.append(x[num], 1)

    # モデルの計算
    X = theta @ y
    z = fun_z(X)
    
    # パラメータの更新
    theta -= eta * (z-t[num]) * y

    if i%N == 0 and i>0:
        print("{}回目".format(i/N))
        fuga = np.zeros(N)
        h = np.zeros(N)
        for j in range(N):
            w = np.append(x[j], 1)
            X = theta @ w
            model_z = fun_z(X)
            if model_z > 0.5:
                fuga[j] = 1
            else:
                fuga[j] = 0
            h[j] = fun_h(t[j], model_z)

        # 交差エントロピーの平均
        H[int(i/N)-1] = np.mean(h)
        print("交差エントロピーの平均：　{}".format(H[int(i/N)-1]))
        
        cnt = 0
        # 識別
        for j in range(N):
            if fuga[j] != lab[j]:
                cnt+=1
        result[int(i/N)-1] = cnt / N * 100
        print("誤識別率：　{}".format(result[int(i/N)-1]))

NNL = np.zeros(NN)
for i in range(NN):
    NNL[i] = i+1

# グラフの出力
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(NNL,result)
ax1.set_xlabel("Number of trials")
ax1.set_ylabel("Error rate [%]")
ax1.set_xlim([0, NN+100])
ax1.set_ylim([0, 100])
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(NNL, H)
ax2.set_xlabel("Number of trials")
ax2.set_ylabel("Mean cross entropy")
ax2.set_xlim([0, NN+100])
ax2.set_ylim([0, 1])
fig.tight_layout()
fig.savefig('result_step02-3/Result(D={}).png'.format(D))
fig.show()