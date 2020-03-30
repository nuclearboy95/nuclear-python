import logging


class TelegramHandler(logging.Handler):
    def __init__(self):
        import ntelegram
        super().__init__()
        self.bot = ntelegram.get_bot('nuclear_bot')

    def emit(self, record):
        msg = self.format(record)
        self.bot.send(msg)
