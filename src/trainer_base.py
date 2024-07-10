class TrainerBase:
    """Base Trainer Class"""

    def __init__(self, config_dir):
        self.config_dir = config_dir

    def train(self):
        raise NotImplementedError("Train method not implemented!")
