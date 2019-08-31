import tensorflow as tf
import numpy as np
import os
import shutil
from collections import Counter
from one import data_set, images_init as data_lib


tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = data_set.TEST_DATA["BATCH_SIZE"]
IMAGE_SIZE = data_set.TEST_DATA["IMAGE_SIZE"]
N_CLASSES = data_set.TEST_DATA["N_CLASSES"]


TEST_DIR = data_set.DATA_DIR["TEST_DATA"] # 测试集存储总目录
MODEL_PATH = data_set.DATA_DIR["MODEL_PATH"] # 模型地址
test_data, test_label = data_lib.get_all_data(TEST_DIR) # 获取数据
NUM_TEST = len(test_label) # 测试集总量


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
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))
# 剪切文件
def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist mv!" % (srcfile))
    else:
        shutil.move(srcfile, dstfile)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))
# 评估模型
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


def test(data):
    model = tf.train.import_meta_graph(MODEL_PATH + ".meta")
    graph = tf.get_default_graph()

    inputs = graph.get_operation_by_name('x-input').outputs[0]
    labels = graph.get_operation_by_name('y-input').outputs[0]
    is_train = graph.get_operation_by_name('is_train').outputs[0]

    # 返回一个list. 但是这里只要第一个参数即可
    pred = tf.get_collection('pred_network')[0]

    with tf.Session(graph=graph) as sess:
        model.restore(sess, MODEL_PATH)
        # 取出测试集合
        x, y = data_lib.get_batch(data[0], data[1], IMAGE_SIZE, IMAGE_SIZE, BATCH_SIZE, BATCH_SIZE * 2, N_CLASSES)
        coord = tf.train.Coordinator() #创建一个线程管理器（协调器）对象
        # 启动tensor的入队线程，可以用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_pred_acc = []
        test_label_acc = []


        for i in range(NUM_TEST // BATCH_SIZE):
            print(NUM_TEST, "------", i * BATCH_SIZE)
            test_x, test_y = sess.run([x, y])
            test_label_acc.append(np.argmax(test_y, 1)) # 标签的值

            # 使用y进行预测
            pred_y = sess.run(tf.nn.softmax(pred, 1), feed_dict={inputs: test_x, labels: test_y, is_train: [False]}) # 替换参数
            test_pred_acc.append(pred_y)

        test_label_acc = np.reshape(test_label_acc, [(NUM_TEST // BATCH_SIZE) * BATCH_SIZE])
        test_pred_acc = np.reshape(test_pred_acc, [(NUM_TEST // BATCH_SIZE) * BATCH_SIZE, N_CLASSES])
        test_label_acc = tf.cast(test_label_acc, dtype=tf.int32)
        test_pred_acc = tf.cast(test_pred_acc, dtype=tf.float32)

        pred_acc = evaluation(test_pred_acc, test_label_acc)
        acc, xyn_estimate, xyn_know = sess.run([pred_acc, test_pred_acc, test_label_acc])
        print("accuracy : ", acc)

        coord.request_stop() # 终止所有线程命令
        coord.join(threads) # 把线程加入主线程，等待threads结束

        ###################################################### 新添加的 ################################################
        Y_estimate = list(np.argmax(xyn_estimate, 1))  # 获取模型预测列表
        Y_know = list(xyn_know)  # 获取已知标签列表
        my_picture = []  # 对应的文件名
        for iio in data[0]:
            my_picture.append(iio[12:].decode('utf-8'))

        False_Positice_Rate = []  # 保存每个种类的假阳性率
        False_Negatice_Rate = []  # 保存每个种类的假阴性率
        Abnormal_Error_Rate = [] # 保留6种错误类型的异常错误率

        for iu in ["0_to_other6", "other6_to_0", "other6_to_other6"]:
            if os.path.exists(iu) == False:  # 判断当前目录是否存在文件夹，不存在就新建
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

        dd = np.sum(cc, axis=1)  # 按行求和
        ee = np.sum(cc, axis=1)  # 按列求和
        all = np.sum(cc)  # 求矩阵所有元素的和

        # 对第0类的情况
        for i in range(7):
            TP = cc[i, i]
            FN = dd[i] - TP
            FP = ee[i] - TP
            TN = all - TP - FN - FP
            if (FP + TN) == 0:
                False_Positice_Rate.append(0)
            else:
                False_Positice_Rate.append(FP / (FP + TN))
            if (TP + FN) == 0:
                False_Negatice_Rate.append(0)
            else:
                False_Negatice_Rate.append(FN / (TP + FN))

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

# 传入模型
test([test_data, test_label])
