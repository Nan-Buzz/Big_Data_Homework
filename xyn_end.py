import os
import shutil
import numpy as np
from collections import Counter

# Windows 下加快 CPU 运行速度
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 查找文件
def search(path, name):
    for root, dirs, files in os.walk(path):  # path 为根目录
        if name in files:
            flag = 1  # 标记找到了文件
            root = str(root)
            return os.path.join(root, name)
    return -1


# 复制文件
def mycopyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist cp!" % (srcfile))
    else:
        shutil.copy(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


# 剪切文件
def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist mv!" % (srcfile))
    else:
        shutil.move(srcfile, dstfile)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))



Y_know = [6, 0, 3, 5, 0, 4]  # 我们标记的种类
Y_estimate = [4, 2, 0, 3, 6, 6]  # 学习得到的种类
my_picture = ['...20170401115302.G1.N11.T11...0 (2).jpg', '...20170410172321.G1.N11.T11...2_fz.jpg', '51a41963-8009-48e4-adf9-ab0f36eccd6520180411.jpg', '22f6063a-be89-4273-ba0e-8c7805277f9720180416.jpg', '0+9d70bb28-0157-467a-86a7-6ca96ef7c8a3.jpg', '0+51a33abf-d9c8-4215-ac79-c7e61a29d6c8.jpg']  # 对应的文件名
False_Positice_Rate = [] # 保存每个种类的假阳性率
False_Negatice_Rate = [] # 保存每个种类的假阴性率
Abnormal_Error_Rate = [] # 保留6种错误类型的异常错误率


for iu in ["0_to_other6", "other6_to_0", "other6_to_other6"]:
    if os.path.exists(iu) == False: # 判断当前目录是否存在文件夹，不存在就新建
        os.makedirs(iu)

for i, j, k in zip(Y_know, Y_estimate, my_picture):
    if i != j:  # 表示识别出错的情况
        srcfile = search("./data/", k)  # 当前目录下找对应文件
        if i == 0:  # 表将0类识别成其他
            mycopyfile(srcfile, "./0_to_other6/" + k)  # 复制文件
        elif j == 0:  # 就其他类识别成0
            mycopyfile(srcfile, "./other6_to_0/" + k)  # 复制文件
        else:  # 其他类识别成其他类
            mycopyfile(srcfile, "./other6_to_other6/" + k)  # 复制文件

# 此处对每个文件夹里面的文件重新编号
for iu in ["./0_to_other6/", "./other6_to_0/", "./other6_to_other6/"]:
    for index, value in enumerate(os.listdir(iu), start=1):
        mymovefile(iu + value, iu + str(index) + '.jpg')


cc = np.zeros((7, 7))
for i, j in zip(Y_know, Y_estimate):
    cc[i, j] += 1

dd = np.sum(cc, axis=1) # 按行求和
ee = np.sum(cc, axis=1) # 按列求和
all = np.sum(cc) # 求矩阵所有元素的和

# 对第0类的情况
for i in range(7):
    TP = cc[i, i]
    FN = dd[i] - TP
    FP = ee[i] - TP
    TN = all - TP - FN - FP
    if (FP + TN) == 0: False_Positice_Rate.append(0)
    else: False_Positice_Rate.append(FP / (FP + TN))
    if (TP + FN) == 0: False_Negatice_Rate.append(0)
    else: False_Negatice_Rate.append(FN/(TP + FN))

print("假阳性率为", False_Positice_Rate)
print("假阴性率为", False_Negatice_Rate)

# 计算异常错误率：
xyn_all_error = Counter(Y_know) # 统计总的图片数据
for i in range(1, 7):
    if xyn_all_error[i] == 0:
        Abnormal_Error_Rate.append(0)
    else:
        Abnormal_Error_Rate.append(cc[i, i] / xyn_all_error[i])
print("异常检测率为", Abnormal_Error_Rate)



