import logging
import tkinter as tk


class CanvasView(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        """Resize self on resize event

        Args:
            event (_type_): the resize event args
        """
        # print(f"event: size={event.width}x{event.height}")
        w = self.master.winfo_width()
        h = self.master.winfo_height()
        # print(f"self.master={self.master}: size={w}x{h}")
        self.height = h
        self.width = w
        # dotw: don't config size or canvas will cover buttons below!
        # self.config(width=self.width, height=self.height)
        # self.logger.info(f"new size={self.width}x{self.height}")
