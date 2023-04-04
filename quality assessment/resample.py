import numpy as np

data = np.loadtxt('D:/bcg_data/data/chenzhihui/bcg_test/DSbcg_sig.txt')    # 导入原始数据集
result = data.resample(rule='6H').interpolate()

np.savetxt('D:/bcg_data/data/chenzhihui/raw_org1000hz.txt',result,fmt="%f",delimiter="/n")