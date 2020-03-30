import logging


class TelegramHandler(logging.Handler):
    def __init__(self):
        import ntelegram
        super().__init__()
        self.bot = ntelegram.get_bot('nuclearbot')

    def emit(self, record):
        msg = self.format(record)
        chat_id = self.bot.get_updates()[-1].message.chat_id
        self.bot.send_message(
            chat_id=chat_id,
            text=msg,
        )
