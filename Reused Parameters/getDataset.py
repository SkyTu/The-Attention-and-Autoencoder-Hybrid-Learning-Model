import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cols = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2',
        'SO2_24h', 'NO2', 'NO2_24h', 'CO', 'CO_24h', 'AT', 'DPT', 'SLP','WSR']


# cols = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h',
#         'O3_8h_24h', 'CO', 'CO_24h', 'AT', 'DPT', 'SLP', 'WD', 'WSR']


class GetDataset(object):
    def __init__(self, year, step, batch_size, data_size, predict_size, split_rate, city="shanghai"):
        """
            :param year: 数值型，使用第几年的数据。
            :param step 每批数据相对于前一批数据移动的步长。
            :param batch_size 一次构建多少批数据。
            :param data_size 预测数据和被预测数据的长度（例如48天历史数据和24天预测数据共需要72天的data_size）
            :param predict_size 预测长度
            :param split_rate 训练集和测试集的分割比率
            :param city: 数据集所在城市
        """
        current_pollutants_dir = 'D:/paper_dataset/classified_dataset/concated_data/'+city+'/'
        self.year = year
        self.day_time = 24  # 一天24小时
        self.data_size = data_size
        self.predict_size = predict_size
        self.step = step
        self.batch_size = batch_size
        self.current_dataset = pd.read_csv(current_pollutants_dir + str(year) + city+'concated_data.csv',
                                           usecols=cols)  # 当前数据
        # self.current_dataset[
        #     ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h',
        #      'O3_8h',
        #      'O3_8h_24h']] = round(self.current_dataset[
        #                                ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2',
        #                                 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h',
        #                                 'O3_8h_24h']])
        self.current_dataset[
            ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2',
             'SO2_24h', 'NO2', 'NO2_24h', 'CO', 'CO_24h']] = round(self.current_dataset[
                                                                       ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h',
                                                                        'SO2',
                                                                        'SO2_24h', 'NO2', 'NO2_24h', 'CO', 'CO_24h']])
        self.PM2_5_mean = self.current_dataset.mean().values[1]
        self.PM2_5_std = self.current_dataset.std().values[1]
        self.current_dataset = (self.current_dataset - self.current_dataset.mean()) / self.current_dataset.std()
        self.split_rate = split_rate  # 划分率
        self.days = len(self.current_dataset) // 24  # 一共有多少天
        self.current_train_dataset = self.current_dataset.values[:round(self.days * self.split_rate) * self.day_time]
        self.test_index = 0

    def train_dataset(self):
        index_begin = 0
        while index_begin + self.data_size + self.step * self.batch_size < len(self.current_train_dataset):
            _x = list()
            _y = list()
            for data_index in range(self.batch_size):
                # _x中存储的是(batch_size,time_step-predict_size,features) _y中存储的是(batch_size,predict_size,features)
                # _x中每次移动step，存入time_step-predict_size长度的内容
                _x.append(self.current_train_dataset[index_begin + data_index * self.step:
                                                     index_begin + data_index * self.step + self.data_size - self.predict_size])
                # _y中每次移动step，存入不在_x中的剩下的predict_size的内容，即预测的值的label
                _y.append(self.current_train_dataset[
                          index_begin + data_index * self.step + self.data_size - self.predict_size:
                          index_begin + data_index * self.step + self.data_size])
            _x = np.array(_x)
            _x = np.squeeze(_x)
            _y = np.squeeze(_y)
            index_begin += self.step
            yield _x, _y.T[1].T
        text_index = index_begin + self.step * self.batch_size
        self.test_index = text_index

    def test_dataset(self):
        current_test_dataset = self.current_dataset.values[self.test_index:]
        index_begin = 0
        while index_begin + self.data_size < len(current_test_dataset):
            _x = list()
            _y = list()
            _x.append(current_test_dataset[index_begin:
                                           index_begin + self.data_size - self.predict_size])
            _y.append(current_test_dataset[
                      index_begin + self.data_size - self.predict_size:
                      index_begin + self.data_size])
            _x = np.array(_x)
            _x = np.squeeze(_x)
            _y = np.squeeze(_y)
            index_begin += self.data_size
            yield _x, _y.T[1].T

    def final_data(self, steps):  # 最终比较图像使用的预测数据
        return self.current_dataset.values[
               self.test_index :self.test_index + self.data_size + (steps-1) * self.predict_size]

    def get_mean_std(self):
        return self.PM2_5_mean, self.PM2_5_std


import matplotlib.pyplot as plt

if __name__ == '__main__':
    gd = GetDataset(year=2018, step=24, batch_size=64, data_size=96, predict_size=24, split_rate=0.7)
    for train_data, train_label in gd.train_dataset():
        pass
    count = 0
    for a, b in gd.test_dataset():
        if count == 0:
            test_data = a
            test_label = b
        count += 1
    c = gd.final_data(24)
    print(type(c))
    plt.plot(np.arange(0, len(c)), c.T[1].T)
    plt.show()
