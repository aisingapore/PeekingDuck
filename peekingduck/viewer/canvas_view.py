# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A resizable canvas for Tkinter
"""

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
