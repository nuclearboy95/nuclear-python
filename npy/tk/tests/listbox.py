from npy.tk import *
from tkinter import *


class MyWindow(Window):
    def set_layout(self):
        with T(widget=Frame(self.root, bg='white')) as frame:
            sb = ScrollableListbox(frame)
            for i in range(20):
                sb.listbox.insert(END, str(i))
            sb.pack()
            sb.subpack()
            frame.pack()


window = MyWindow()
window.main()
