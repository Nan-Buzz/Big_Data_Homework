'''
模型数据中心
'''


TRAIN_DATA = {"BATCH_SIZE":16,              #训练批度
              "LEARNING_RATE_BASE":0.0001,  #基础学习率
              "TRAINING_STEPS":50000,       #训练次数
              "IMAGE_SIZE":64,              #图像大小
              "NUM_CHANNELS":3,             #图像深度
              "N_CLASSES":7}                #分类数目



TEST_DATA = {"BATCH_SIZE":20,               #测试批度
             "IMAGE_SIZE":64,               #图像大小
             "N_CLASSES":7}                 #分类数目



DATA_DIR = {"ROOT_DATA":"data",             #数据集的总目录
            "TRAIN_DATA":"train_data",      #训练集存储总目录
            "TEST_DATA":"test_data",        #测试集存储总目录
            "DATA_NAME":"data_set",         #数据字典文本
            "MODEL_PATH":'model/mod'}       #模型地址
