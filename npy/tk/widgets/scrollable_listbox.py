from tkinter import *


__all__ = ['ScrollableListbox']


class ScrollableListbox(Frame):
    def __init__(self, master, orient='vertical'):
        super().__init__(master)

        self.orient = orient
        self.listbox = Listbox(self)
        self.scrollbar = Scrollbar(self, orient=orient, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=self.scrollbar.set)

    def subpack(self):
        if self.orient == 'vertical':
            self.listbox.pack(side='left', fill='y')
            self.scrollbar.pack(side='right', fill='y')
        else:
            self.listbox.pack(side='top', fill='x')
            self.scrollbar.pack(side='bottom', fill='x')

