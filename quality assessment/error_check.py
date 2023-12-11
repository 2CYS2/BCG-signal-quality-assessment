"""
 @Author: cys
 @Email: 2022024904@m.scnu.edu.cn
 @FileName: error_check.py
 @Function: 绘制错判的片段进行纠正
 @DateTime: 2023/5/2 14:23
 @SoftWare: PyCharm
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import Data_utils as DAU


def data(data_root):
    """
    读取数据并转化为数组
    :param data_root:
    :param saveroot:
    :return:
    """
    BCG_sync_, Jpeaks_sync_, Jpeak_DL_, Rpeaks_sync_ = DAU.read_data(data_root)

    BCG = BCG_sync_
    Jpeak_AI = Jpeaks_sync_
    Jpeak_DL = Jpeak_DL_
    Rpeaks = Rpeaks_sync_

    return BCG, Jpeak_AI, Jpeak_DL, Rpeaks


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


def artifact(artifact_data, total_time):
    """
    创建2个与BCG信号等长的信号，一个体动区域为1，其余为0；另一个体动与BCG一样，其余为nan
    :param artifact_data: 体动文件
    :param total_time: BCG总时长
    :return:
    """
    atf_if = np.zeros(total_time)
    atf_nan = np.full([total_time, ], np.nan)
    for i in range(0, int(len(artifact_data) / 4)):
        atf_start = int(artifact_data[4 * i + 2])
        atf_end = int(artifact_data[4 * i + 3])
        if atf_start > atf_end:  # 若存在此情况，即将两者调换
            print("atf_start = ", atf_start)
            print("atf_end = ", atf_end)
            print("终止点小于起始点")
            atf_end, atf_start = atf_start, atf_end
        atf_if[atf_start: atf_end] = 1
        atf_nan[atf_start: atf_end] = BCG[atf_start: atf_end]

    return atf_if, atf_nan


def artifact_del(artifact_data, jpeaks1, jpeaks2):
    """
    根据体动文件去除处于体动区间的J峰坐标
    :param artifact_data: 体动数据文件
    :param jpeaks: J峰文件
    :return: 去除体动区间内的J峰后剩余的J峰集合
    """
    for i in range(0, int(len(artifact_data) / 4)):
        min = jpeaks1 > artifact_data[4 * i + 2]
        max = jpeaks1 < artifact_data[4 * i + 3]
        rst = min & max
        j_del = np.where(rst == True)  # 寻找错判J峰
        jpeaks1 = np.delete(jpeaks1, j_del, axis=0)

        min_r = jpeaks2 > artifact_data[4 * i + 2]
        max_r = jpeaks2 < artifact_data[4 * i + 3]
        rst_r = min_r & max_r
        r_del = np.where(rst_r == True)  # 寻找错判J峰
        jpeaks2 = np.delete(jpeaks2, r_del, axis=0)

    jpeaks1_new = jpeaks1.astype(int)
    jpeaks2_new = jpeaks2.astype(int)

    return jpeaks1_new, jpeaks2_new


def GRADE(grade, scale):
    """
    将对应等级的字符转化为数字
    :param grade:
    :return:
    """
    grade_num = 0
    if scale == 30:
        if grade == 'a1':
            grade_num = 1
        elif grade == 'a2':
            grade_num = 2
        elif grade == 'b1':
            grade_num = 3
        elif grade == 'b2':
            grade_num = 4
        elif grade == 'c':
            grade_num = 5
    elif scale == 10:
        if grade == 'a':
            grade_num = 1
        elif grade == 'b':
            grade_num = 2
        elif grade == 'c':
            grade_num = 3

    return grade_num


def picture_show(BCG, Jpeaks1, Jpeaks2, grade, scale, i, filename, result_path, Jpeak_real, art_nan_cut):
    """
    绘图
    :param BCG:
    :param Jpeaks1:
    :param Jpeaks2:
    :param grade:
    :param model:
    :param scale:
    :param i:
    :return:
    """
    size = 10  #35
    plt.figure(figsize=(8, 6))    #figsize=(30, 20), figsize=(8, 6)
    plt.title("NO.%d      Quality:%s" % (i, grade), fontsize=size)  # 当前显示片段对应的编号和质量等级
    grade = grade.split("\n")[0]
    if scale == 10:
        if grade == 'a':
            plt.plot(BCG, linestyle='-', label='BCG')
        elif grade == 'b':
            plt.plot(BCG, linestyle='-', label='BCG', color='g')
        else:
            plt.plot(BCG, linestyle='-', label='BCG', color='black')
    else:
        if grade == 'a1':
            plt.plot(BCG, linestyle='-', label='BCG', color='b')
        elif grade == 'a2':
            plt.plot(BCG, linestyle='-', label='BCG', color='skyblue')
        elif grade == 'b1':
            plt.plot(BCG, linestyle='-', label='BCG', color='forestgreen')
        elif grade == 'b2':
            plt.plot(BCG, linestyle='-', label='BCG', color='lime')
        else:
            plt.plot(BCG, linestyle='-', label='BCG', color='black')
    plt.scatter(Jpeak_real, BCG[Jpeak_real], s=50, marker='^')  # 绘制实际J峰，可注释s=500
    for k in range(len(Jpeak_real)):
        plt.text(Jpeak_real[k], BCG[Jpeak_real[k]] + 10, (Jpeak_real[k]))
    plt.scatter(Jpeaks1, BCG[Jpeaks1], s=40, marker='o')    # s=400
    plt.scatter(Jpeaks2, BCG[Jpeaks2], s=40, marker='x')
    plt.plot(art_nan_cut, 'r', label='Artifact')
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.legend(['BCG', 'Jpeaks_real', 'Jpeaks_DL', 'Jpeaks_TM'], loc='lower right', fontsize=size)  # 给曲线都上图例

    plt.savefig(os.path.join(saveroot, str(filename)) + '/' + str(i) + '.png')
    plt.close('all')  # 关闭所有打开的图片，防止报错
    # if show_if:
    #     plt.show(block=False)
    # while True:
    #     if plt.waitforbuttonpress():
    #         plt.close()
    #         break

if __name__ == '__main__':
    # --------------------------------------参数设置------------------- -------------------
    fs = 1000
    scale = 10  # 信号质量评估的时间尺度
    ModelLength = 850  # 根据统计NN间期均值987、标准差120而定
    show_if = 1
    saveroot = r"E:\研一\信号质量相关性统计\小论文\结果汇总5.17\9.18更新\讨论/"
    error_path = r"E:\研一\信号质量评估\result10.13\normalization\留一法(DL)--特征级联(按特征定义排序)\error_index/"
    data_root = r"E:\BCG_data"
    TM_file = r"E:\BCG_data\备注\model_match_result/"
    # filenames = np.loadtxt(r"E:\研一\信号质量评估\file_num.txt", encoding='utf-8')

    filenames = [1009]
    for filename in filenames:
        print("--------------------------------------正在处理%s.txt--------------------------------------" % filename)
        filename = str(int(filename))
        result_path = os.path.join(saveroot, str(filename))  # 创建一个文件夹用于保存错判片段图像
        mkdir(result_path)

        quality_file = open(os.path.join(data_root, filename + '/' + '质量' + '/' + 'SQ_label1_10s.txt'), 'r+',
                            encoding='utf-8')
        artifact_data = np.loadtxt(os.path.join(data_root, filename + '/' + '体动' + '/' + 'Artifact_a.txt'),
                                   encoding='utf=8')
        Jpeak_TM = np.loadtxt(os.path.join(TM_file, filename + 'predict_J.txt'), encoding='UTF-8')
        # Jpeak_TM = np.loadtxt(r"E:\BCG_data\551\Align\551predict_J.txt", encoding='utf-8')

        BCG, Jpeak_AI, Jpeak_DL, Rpeaks = data(os.path.join(data_root, filename + '/' + 'Align/'))

        # Jpeak_DL = np.loadtxt(r"E:\研一\信号质量相关性统计\小论文\结果汇总5.17\9.18更新\result2\10.19异常样本分析\A级图片\深度重定位结果\551Jpeak_DL.txt", encoding='utf-8').astype('int64')
        # Jpeak_AI = np.loadtxt(r"E     :\BCG_data\551\Align\J峰标签更正\Jpeaks_sync.txt", encoding='utf-8').astype('int64')
        # Jpeak_AI = np.sort(Jpeak_AI)
        # np.savetxt(r"E:\BCG_data\551\Align\J峰标签更正\Jpeaks_sync.txt", Jpeak_AI, fmt='%s')
        Jpeak_AI_new = []
        # for k in range(len(Jpeak_AI)):
        #     Jpeak_AI[k] = int(Jpeak_AI[k])

        # error_index = np.loadtxt(error_path + filename + '.txt', encoding='utf=8')
        quality_list = quality_file.readlines()  # 读取所有行的元素并返回一个列表

        # 1、绘制C级信号片段
        error_index = []
        for j in range(0, len(quality_list)):
            # error_index.append(j + 1)
            if quality_list[j].split('\n')[0] == 'a':
                error_index.append(j + 1)
        # print(error_index)

        # 2、绘制指定片段
        error_index = [103]
        # error_index = []
        # # data = np.loadtxt(os.path.join(r"E:\研一\信号质量评估\result10.13\normalization\留一法(DL)--特征级联(特征贡献度排序)\test&pre/", filename + '.txt'), encoding='utf-8')
        # for k in range(len(data[:, 0])):
        #     if data[k, 0] == 3 and data[k, 1] == 2:
        #         error_index.append(k+1)

        print("质量片段总数为：", len(quality_list))
        # artifact_if = artifact(artifact_data, len(BCG))  # 将体动区间内的点置为1
        artifact_if, atf_nan = artifact(artifact_data, len(BCG))  # 将体动区间内的点置为1
        # Jpeak_DL, Jpeak_TM = artifact_del(artifact_data, Jpeak_DL, Jpeak_TM)
        Jpeak_AI1, Jpeak_TM = artifact_del(artifact_data, Jpeak_AI, Jpeak_TM)  # 处理实际J峰，可注释

        # chance = int(input("请输入纠正方式（1为浏览，0为指定）："))
        # if chance == 0:
        #     i = int(input("请输入指定纠正的片段标编号："))
        # else:
        for i in error_index:  # 信号切片处理
            print("第", i, "个片段")
            i = int(i - 1)
            grade_num = GRADE(quality_list[i].split('\n')[0], scale)  # 将对应等级转换为数字，便于处理
            # atf_result = 0
            # if scale == 10:
            #     if sum(artifact_cut == 1):  # 判断该片段是否含有体动，且根据不同时间尺度做不同的处理
            #         atf_result = 1
            # elif scale == 30:
            #     atf_result = sum(artifact_cut == 1) / len(BCG_cut)
            # Jpeak_AI_cut_index = np.where(((Jpeak_AI > i * scale * fs + 1) * (Jpeak_AI < (i + 1) * scale * fs - 1)) == True)  # 该段信号对应的实际J峰集合，可注释
            # Jpeak_AI = np.delete(Jpeak_AI, Jpeak_AI_cut_index, axis=0)

            Jpeak_AI_cut = Jpeak_AI[(Jpeak_AI > i * scale * fs + 1) * (
                    Jpeak_AI < (i + 1) * scale * fs - 1)] - i * scale * fs  # 该段信号对应的实际J峰集合，可注释
            Jpeak_DL_cut = Jpeak_DL[(Jpeak_DL > i * scale * fs + 1) * (
                    Jpeak_DL < (i + 1) * scale * fs - 1)] - i * scale * fs  # 该段信号对应的深度学习定位J峰集合
            Jpeak_TM_cut = Jpeak_TM[(Jpeak_TM > i * scale * fs + 1) * (
                    Jpeak_TM < (i + 1) * scale * fs - 1)] - i * scale * fs  # 该段信号对应的模板匹配定位J峰集合
            BCG_cut = BCG[i * scale * fs:(i + 1) * scale * fs - 1]  # 截取第i+1段信号
            art_nan_cut = atf_nan[i * scale * fs:(i + 1) * scale * fs - 1]  # 截取该片段的体动

            # for a in range(len(Jpeak_AI_cut)):
            #     if Jpeak_AI_cut[a] > 9960 or Jpeak_AI_cut[a] < 40:
            #         pass
            #     else:
            #         Jpeak_AI_cut[a] = np.argmax(BCG_cut[Jpeak_AI_cut[a]-40:Jpeak_AI_cut[a]+40]) + Jpeak_AI_cut[a] - 40
            #     Jpeak_AI_new.append(Jpeak_AI_cut[a] + i * scale * fs)
            # # #
            # for a in range(len(Jpeak_AI_cut)):
            #     print(Jpeak_AI_cut[a] + i * scale * fs)

            picture_show(BCG_cut, Jpeak_DL_cut, Jpeak_TM_cut, quality_list[i], scale, i + 1, filename, result_path,
                         Jpeak_AI_cut, art_nan_cut)

        # np.savetxt(r"E:\BCG_data\551\Align\J峰标签更正\更新的J峰标签_new.txt", Jpeak_AI_new, fmt='%s')
#