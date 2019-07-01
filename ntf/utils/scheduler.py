
__all__ = ['Scheduler']


class Scheduler:
    def __init__(self, start_lr=1e-3, max_decay=4, patience=5, decay=0.3):
        self.start_lr = start_lr
        self.max_decay = max_decay
        self.patience = patience
        self.decay = decay

        self.patience_count = 0
        self.decay_count = 0

        self.last_value = float('inf')

    def on_step(self, value):
        if self.last_value > value:  # improved
            self.patience_count = 0
            self.last_value = value

        else:
            self.patience_count += 1
            if self.patience_count >= self.patience:
                self.patience_count = 0
                self.decay_count += 1

    @property
    def lr(self):
        if self.decay_count > self.max_decay:
            return None
        else:
            return self.start_lr * (self.decay ** self.decay_count)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
