def bind_left_click(widget, callback):
    widget.bind('<Button-1>', callback)


def bind_right_click(widget, callback):
    widget.bind('<Button-3>', callback)


def bind_scroll_up(widget, callback):
    widget.bind('<Button-4>', callback)


def bind_scroll_down(widget, callback):
    widget.bind('<Button-5>', callback)


def bind_left_drag(widget, callback):
    widget.bind('<B1-Motion>', callback)


def bind_right_drag(widget, callback):
    widget.bind('<B3-Motion>', callback)


def bind_key_press(widget, callback):
    widget.bind('<Key>', callback)


def bind_listbox_select(widget, callback):
    widget.bind('<<ListboxSelect>>', callback)
