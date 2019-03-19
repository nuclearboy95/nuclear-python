from npy.tk import *
from tkinter import *


class MyWindow(Window):
    def set_layout(self):
        with T(widget=Canvas(self.root)) as canvas:
            with T(widget=Canvas(canvas)) as canvas2:
                canvas2.create_oval(10, 10, 15, 15, fill='blue')
                canvas2.pack()

            canvas.create_oval(0, 0, 10, 10, fill='red')
            canvas.pack()


window = MyWindow()
window.main()
