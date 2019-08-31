import tensorflow as tf
import os
import numpy as np
from one import data_set, images_init as data_lib, VGG_models as VGG_model

# Windows 下加快 CPU 运行速度
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = data_set.TRAIN_DATA["BATCH_SIZE"]
LEARNING_RATE_BASE = data_set.TRAIN_DATA["LEARNING_RATE_BASE"]
TRAINING_STEPS = data_set.TRAIN_DATA["TRAINING_STEPS"]
IMAGE_SIZE = data_set.TRAIN_DATA["IMAGE_SIZE"]
NUM_CHANNELS = data_set.TRAIN_DATA["NUM_CHANNELS"]
N_CLASSES = data_set.TRAIN_DATA["N_CLASSES"]

# 分为Train集合的占比
PROPORTION = 0.8
# 数据集的总目录
DIR_PATH = data_set.DATA_DIR["ROOT_DATA"]
# 训练集存储总目录
TRAIN_DIR = data_set.DATA_DIR["TRAIN_DATA"]


def train(train):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, N_CLASSES], name='y-input')
    is_train = tf.placeholder(tf.bool, [1], name="is_train")

    y = VGG_model.VGG11_inference(x, N_CLASSES, is_train[0])
    loss = VGG_model.losses(y, tf.argmax(y_, 1))
    acc = VGG_model.evaluation(y, tf.argmax(y_, 1))
    corrects = VGG_model.estimate(y, tf.argmax(y_, 1))
    train_step = VGG_model.trainning(loss, LEARNING_RATE_BASE, BATCH_SIZE, len(train[0]))


    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    tf.add_to_collection('pred_network', y)
    saver = tf.train.Saver(tf.all_variables())

    test_pred_acc = []
    test_label_acc = []
    pred = tf.get_collection('pred_network')[0]

    with tf.Session() as sess:
        xs, ys = data_lib.get_shuffle_batch(train[0], train[1], IMAGE_SIZE, IMAGE_SIZE, BATCH_SIZE, BATCH_SIZE * 2, N_CLASSES)
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            xx, yy = sess.run([xs, ys])
            test_label_acc.append(np.argmax(yy, 1))
            _, loss_value, acceval, pred_y = sess.run([train_op, loss, acc, tf.nn.softmax(pred, 1)], feed_dict={x: xx, y_: yy, is_train: [True]})
            test_pred_acc.append(pred_y)

            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value), "acc : ", acceval)
                if loss_value is not None:
                    saver.save(sess, 'model/mod')


        test_label_acc = np.reshape(test_label_acc, [TRAINING_STEPS * BATCH_SIZE])
        test_pred_acc = np.reshape(test_pred_acc, [TRAINING_STEPS * BATCH_SIZE, N_CLASSES])
        test_label_acc = tf.cast(test_label_acc, dtype=tf.int32)
        test_pred_acc = tf.cast(test_pred_acc, dtype=tf.float32)

        pred_acc = VGG_model.evaluation(test_pred_acc, test_label_acc)
        corr = tf.nn.in_top_k(test_pred_acc, test_label_acc, 1)
        acc, pred_xyn, label_xyn, corr_xyn = sess.run([pred_acc, test_pred_acc, test_label_acc, corr])
        print("accuracy : ", acc)
        print(np.argmax(pred_xyn, 1))
        print(label_xyn, type(label_xyn))
        print(corr_xyn, type(corr_xyn))

        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    if os.path.exists(TRAIN_DIR) is False:
        data_lib.split_physical(DIR_PATH, PROPORTION)
    train_data, train_label = data_lib.get_all_data(TRAIN_DIR)
    train([train_data, train_label])


if __name__ == '__main__':
    main()
