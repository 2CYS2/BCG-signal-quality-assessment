"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: test2.py
 @Function: 
 @DateTime: 2022/11/13 11:01
 @SoftWare: PyCharm
"""
from itertools import groupby
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    filepath = r"E:\BCG_data/"
    result3_num = np.loadtxt(r"E:\研一\信号质量相关性统计\Python\result\detect_performance\result2\quality.txt", encoding='utf-8')
    num, A1, A2, B1, B2, C_30, A, B, C_10 = [], [], [], [], [], [], [], [], []
    for filename in result3_num:
        filename = str(int(filename))
        data_30 = np.loadtxt(r'E:\BCG_data\969\质量\SQ_label1_30s.txt', dtype=str)
        data_root = os.path.join(filepath, filename)
        grade_data = np.loadtxt(os.path.join(os.path.join(data_root, '质量'), 'SQ_label2_30s.txt'), dtype=str)    # 质量标签
        grade_data_10 = np.loadtxt(os.path.join(os.path.join(data_root, '质量'), 'SQ_label2_10s.txt'), dtype=str)    # 质量标签
        filename = int(filename)
        print(grade_data[2])

        num.append(filename)
        A1.append(int(grade_data[2]))
        A2.append(int(grade_data[4]))
        B1.append(int(grade_data[6]))
        B2.append(int(grade_data[8]))
        C_30.append(int(grade_data[10]))

        A.append(int(grade_data_10[2]))
        B.append(int(grade_data_10[4]))
        C_10.append(int(grade_data_10[6]))

    result_10 = list(zip(num, A1, A2, B1, B2, C_30, A, B, C_10))
    with open(r'E:\研一\信号质量相关性统计\Python\result\detect_performance\result2\quality_data.txt', 'a+') as f:
         np.savetxt(f, result_10,  fmt='%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t', delimiter='\t')