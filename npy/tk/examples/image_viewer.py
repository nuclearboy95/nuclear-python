from ..window_class import *
from tkinter import *
from tkinter import filedialog
from scipy.misc import imread
from PIL import Image, ImageTk
import os


__all__ = ['ImageViewer']


class ImageViewer(Window):
    def __init__(self):
        super().__init__()

        self.image_dir = None
        self.image_paths = []
        self.image_path = None

        self.img_tk = None
        self.img_created = None

        self.canvas_image = None

    def set_layout(self):
        self.set_geometry(w=1200, h=800)
        self.set_title('Image Viewer')

        with T(widget=Frame(self.root, relief='solid', width=1200, height=700)) as frame_image:
            with T(widget=Canvas(frame_image, width=1200, height=700, bd=0, highlightthickness=0, bg='white')) as canvas_image:
                self.canvas_image = canvas_image
                self.canvas_image.pack()

            frame_image.pack()

        with T(widget=Frame(self.root, relief='solid', width=1200, height=100, bg='gray')) as frame_bar:
            button = Button(frame_bar, text='Select', command=self.ask_image_path)
            button.pack()

            frame_bar.pack()

    def ask_image_path(self):
        image_path_selected = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select file",
            filetypes=(("all files", "*.*"),)
        )

        if image_path_selected and os.path.isfile(image_path_selected):
            from ntf import path_listdir
            self.image_path = image_path_selected
            self.image_dir = os.path.dirname(self.image_path)
            image_paths = path_listdir(self.image_dir)
            self.image_paths = list(filter(self.is_image_file, image_paths))

            image = imread(self.image_path)
            self.display_image(image)

    def display_image(self, image):
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(image))
        if self.img_created is not None:
            self.canvas_image.delete(self.img_created)

        self.img_created = self.canvas_image.create_image(600, 350, image=self.img_tk)

    @staticmethod
    def is_image_file(fpath):
        return os.path.splitext(fpath)[1].lower() in ['.png', '.jpeg']
