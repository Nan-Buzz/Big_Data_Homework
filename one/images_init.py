import tensorflow as tf
import os
import shutil
from one import data_set

# 数据集的总目录
DIR_PATH = data_set.DATA_DIR["ROOT_DATA"]
# 训练集存储总目录
TRAIN_DIR = data_set.DATA_DIR["TRAIN_DATA"]
# 测试集存储总目录
TEST_DIR = data_set.DATA_DIR["TEST_DATA"]
# 数据字典文本
DATA_NAME = data_set.DATA_DIR["DATA_NAME"]


# 字典读取
def read_dic(file_path):
    f = open(file_path, 'r')
    a = f.read()
    dict_name = eval(a)
    f.close()
    return dict_name


# 字典存储
def save_dic(dic, file_path):
    f = open(file_path, 'w')
    f.write(str(dic))
    f.close()


# 读取所有文件
def file_name(file_dir):
    labels = []
    files = []
    for root, dirs, file in os.walk(file_dir):
        labels.extend(dirs)
        files.extend(file)
        # 当前目录下与其子目录下的所有文件名，当前目录下的文件夹名
    return files, labels


# 文件逻辑分比
def split_logic(file_dir, prop):
    _, labels = file_name(file_dir)
    images = {}
    for label in labels:
        images[label], _ = file_name(file_dir + "/" + label)
    train_data = {}
    test_data = {}
    for label in labels:
        train = []
        test = []
        num = 0
        for image in images[label]:
            if len(images[label]) * prop > num:
                train.append(image)
            else:
                test.append(image)
            num += 1
        train_data[label] = train
        test_data[label] = test
    del _, labels, images
    # 输出格式：{LABELS：NAME}
    return train_data, test_data


# 文件物理分比
def split_physical(dir_path, proportion):
    train_data, test_data = split_logic(dir_path, proportion)
    if os.path.exists(TRAIN_DIR) is False:
        os.mkdir(TRAIN_DIR)
    if os.path.exists(TEST_DIR) is False:
        os.mkdir(TEST_DIR)
    for label in train_data.keys():
        print("Train file---", label)
        if os.path.exists(TRAIN_DIR + '/' + label) is False:
            os.mkdir(TRAIN_DIR + '/' + label)
        for data in train_data[label]:
            old_file_dir = DIR_PATH + '/' + label + '/' + data
            new_file_dir = TRAIN_DIR + '/' + label + '/' + data
            shutil.copy(old_file_dir, new_file_dir)
    save_dic(train_data, TRAIN_DIR + '/' + DATA_NAME)
    for label in test_data.keys():
        print("Test file---", label)
        if os.path.exists(TEST_DIR + '/' + label) is False:
            os.mkdir(TEST_DIR + '/' + label)
        for data in test_data[label]:
            old_file_dir = DIR_PATH + '/' + label + '/' + data
            new_file_dir = TEST_DIR + '/' + label + '/' + data
            shutil.copy(old_file_dir, new_file_dir)
    save_dic(test_data, TEST_DIR + '/' + DATA_NAME)
    return train_data, test_data


# 随机批量处理
def get_shuffle_batch(image, label, image_w, image_h, batch_size, capacity, n_classes):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_images(image, [image_w, image_h], method=0)

    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=capacity, min_after_dequeue=capacity // 2)

    label_batch = tf.reshape(label_batch, [batch_size])
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    return image_batch, label_batch


# 标准批量处理
def get_batch(image, label, image_w, image_h, batch_size, capacity, n_classes):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_images(image, [image_w, image_h], method=0)

    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    label_batch = tf.one_hot(label_batch, depth=n_classes)

    return image_batch, label_batch


def read_image_tensor(image_dir):
    image = tf.gfile.FastGFile(image_dir, 'rb').read()
    image = tf.image.decode_jpeg(image)  # 图像解码
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)  # 改变图像数据的类型
    return image


# 传入TRAIN_DIR
def get_all_data(file_dir):
    train_data = read_dic(file_dir + '/' + DATA_NAME)
    data = {}
    for label in train_data.keys():
        for image_dir in train_data[label]:
            image = file_dir + '/' + label + '/' + image_dir
            data[image] = int(label)
    del train_data
    return list(data.keys()), list(data.values())

# #分割数据集
# #split_physical(DIR_PATH,PROPORTION)
# #读取所有文件信息
# image,labels = get_all_data(TEST_DIR)
# print(get_all_data(TEST_DIR))
# print(get_all_data(TRAIN_DIR))
# #分批读取
# with tf.Session() as sess:
#     data , label = get_batch(image,labels,224,224,64,len(image),7)
#     tf.global_variables_initializer().run()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     while True :
#         xx, yy = sess.run([data, label])
#         print(yy)
#     coord.request_stop()
#     coord.join(threads)
