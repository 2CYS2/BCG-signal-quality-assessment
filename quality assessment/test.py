"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: test.py
 @Function: 按分类器将其相应的分类性能归整到同一个文件
 @DateTime: 2022/11/12 20:38
 @SoftWare: PyCharm
"""
import os
import numpy as np

if __name__ == '__main__':
    filepath = r"E:\研一\信号质量评估\Result_new\assessment\30s/"
    filenames = os.listdir(filepath)
    RF, DT, KNN, XG, NUM = [], [], [], [], []
    for filename in filenames:
        result = np.loadtxt(os.path.join(filepath, filename), encoding='utf-8')
        num = filename.split('.txt')[0]
        print(num)
        NUM.append(num)
        RF.append(result[0, :])
        DT.append(result[1, :])
        KNN.append(result[2, :])
        XG.append(result[3, :])


    print(NUM)
    # result_RF = list(zip(RF))
    with open(os.path.join(filepath, 'RF.txt'), 'a+') as f:
            np.savetxt(f, RF, fmt='%f', delimiter='\t')
    # result_DT = list(zip(DT))
    with open(os.path.join(filepath, 'DT.txt'), 'a+') as f:
            np.savetxt(f, DT, fmt='%f', delimiter='\t')
    # result_KNN = list(zip(KNN))
    with open(os.path.join(filepath, 'KNN.txt'), 'a+') as f:
            np.savetxt(f, KNN, fmt='%f', delimiter='\t')
    # result_XG = list(zip(XG))
    with open(os.path.join(filepath, 'XG.txt'), 'a+') as f:
            np.savetxt(f, XG, fmt='%f', delimiter='\t')
