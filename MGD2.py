import numpy as np

def getData(nclass, D, seed = None):

    assert nclass == 2 or nclass == 3

    if seed != None:
        np.random.seed(seed)

    hoge0 , hoge1, hoge2 = np.zeros(D), np.zeros(D), np.zeros(D)
    for i in range(D):
        hoge0[i] = 0.3
        hoge1[i] = 0.7 - (i*0.1)
        if i%2==0:
            hoge2[i] = 0.3
        else:
            hoge2[i] = 0.7
        
    # D次元の spherical な正規分布3つからデータを生成
    X0 = 0.10 * np.random.randn(200, D) + hoge0
    X1 = 0.10 * np.random.randn(200, D) + hoge1
    X2 = 0.05 * np.random.randn(200, D) + hoge2

    # それらのラベル用のarray
    lab0 = np.zeros(X0.shape[0], dtype = int)
    lab1 = np.zeros(X1.shape[0], dtype = int) + 1
    lab2 = np.zeros(X2.shape[0], dtype = int) + 2

    # x （入力データ）, label （クラスラベル）, t（教師信号） をつくる
    if nclass == 2:
        x = np.vstack((X0, X1))
        label = np.hstack((lab0, lab1))
        t = np.zeros(x.shape[0])
        t[label == 1] = 1.0
    else:
        x = np.vstack((X0, X1, X2))
        label = np.hstack((lab0, lab1, lab2))
        t = np.zeros((x.shape[0], nclass))
        for ik in range(nclass):
            t[label == ik, ik] = 1.0

    return x, label, t

if __name__ == '__main__':
        
    K = 2
    D = 3
    
    x, lab, t = getData(K, D)

    print(x.shape)