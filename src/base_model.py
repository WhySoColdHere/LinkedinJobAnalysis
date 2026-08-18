class Model:
    def __init__(self, **kwargs):
        self.x_train = kwargs['x_train']
        self.x_test = kwargs['x_test']
        self.y_train = kwargs['y_train']
        self.y_test = kwargs['y_test']
        self.model = None

    def run(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__} instance"
