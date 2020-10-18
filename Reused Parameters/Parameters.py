class Parameters(object):
    def __init__(self, epochs=50, split_rate=0.7, l2_rate=0.05, data_size=96, predict_size=24, node_size=32):
        """
        :param epochs: Model training epochs
        :param split_rate: The split rate of train set and test set
        :param l2_rate: L2 regularization value
        :param data_size: Overall time steps of a batch
        :param predict_size: Time steps of predict period
        :param node_size: Node size of the model LSTM or GRU
        """
        self.epochs = epochs
        self.split_rate = split_rate
        self.l2_rate = l2_rate
        self.features = 15
        self.data_size = data_size
        self.predict_size = predict_size
        self.validate_num = 15
        self.node_size = node_size
        self.train_size = self.data_size - self.predict_size


if __name__ == '__main__':
    p = Parameters()
    print(p.data_size)
    q = Parameters(data_size=100)
    print(q.data_size)
