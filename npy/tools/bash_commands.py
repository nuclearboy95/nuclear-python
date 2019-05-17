import subprocess


__all__ = ['ls', 'lsl']


class BashCommand:
    def __init__(self, commands):
        self.commands = commands

    def evaluate(self):
        p = subprocess.run(self.commands, stdout=subprocess.PIPE)
        output = p.stdout
        if output[-1] == 10:  # \n
            output = output[:-1]
        return output.decode()

    def __str__(self):
        return self.evaluate()

    def __repr__(self):
        return self.evaluate()


ls = BashCommand(['ls'])
lsl = BashCommand(['ls', '-l'])
