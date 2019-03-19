from npy.tk import *
from tkinter import *
import tkinter
import tkinter.ttk


class MyWindow(Window):
    def set_layout(self):
        with T(widget=tkinter.ttk.Notebook(self.root, width=400, height=300)) as notebook:

            with T(widget=Frame(self.root)) as frame1:
                label = Label(frame1, text='Content of frame1')
                label.pack()
                notebook.add(frame1, text='page1')

            with T(widget=Frame(self.root)) as frame2:
                label = Label(frame2, text='Content of frame2')
                label.pack()
                notebook.add(frame2, text='page2')

            notebook.pack()


window = MyWindow()
window.main()
