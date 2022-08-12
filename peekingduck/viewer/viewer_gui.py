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
PeekingDuck Viewer GUI Creation Code
"""

# Technotes:
#   using __future__ and TYPE_CHECKING works with pylint 2.10.x but fails with 2.7.x
#
# from __future__ import annotations
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from peekingduck.player.viewer import Viewer
#
# Did not import Viewer from peekingduck.viewer.viewer due to pylint 2.7 complaining
# about circular import.

from pathlib import Path
import tkinter as tk
from tkinter import ttk
from peekingduck.viewer.viewer_utils import load_image
from peekingduck.viewer.single_column_view import SingleColumnPlayListView

LOGO: str = "PeekingDuckLogo.png"
MIN_HEIGHT: int = 768
MIN_WIDTH: int = 1024
WIN_HEIGHT: int = 960
WIN_WIDTH: int = 1280
BTN_WIDTH_SPAN = 1


def create_window(viewer) -> None:  # type: ignore
    """Create PeekingDuck Viewer window with the following structure:
        +-+-------------------------------+-+
        | |  Logo       Name              | |
        | +-------------------------------+ |
        | |                               | |
        | |                               | |
        | |     Video               Play  | |
        | |     image               list  | |
        | |                               | |
        | |                               | |
        | +-------------------------------+ |
        | |           Controls            | |
        | |           StatusBar           | |
        +-+-------------------------------+-+

    Args:
        viewer (Viewer): PeekingDuck Viewer object

            Required event handlers
                on_exit()
                on_keypress()
                on_resize()
                on_add_pipeline()
                on_delete_pipeline()
                on_play_pipeline()
    """
    root = tk.Tk()
    root.title("PeekingDuck Viewer")
    # bind event handlers
    root.wm_protocol("WM_DELETE_WINDOW", viewer.on_exit)
    root.bind("<Configure>", viewer.on_resize)
    root.bind("<Key>", viewer.on_keypress)
    root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
    root.minsize(MIN_WIDTH, MIN_HEIGHT)
    root.update()  # force update before mainloop() to get correct size
    viewer.root = root  # save main window
    # Tk technotes: Need to create footer before body to ensure footer controls do not
    #               get covered when image is zoomed in
    create_side_margins(viewer)
    create_header(viewer)
    create_footer(viewer)
    create_body(viewer)
    viewer.btn_hide_show_playlist_press()  # first toggle to hide playlist


def create_header(viewer) -> None:  # type: ignore
    """Create header with PeekingDuck logo and pipeline name

    Args:
        viewer (Viewer): PeekingDuck Viewer object
    """
    header_frm = ttk.Frame(viewer.root, name="header_frm")
    header_frm.pack(side=tk.TOP, fill=tk.X)
    viewer.tk_header_frm = header_frm
    # row 0: logo (left)
    logo_path = Path(__file__).parent / LOGO
    viewer.img_logo = load_image(str(logo_path), resize_pct=0.10)
    logo = tk.Label(header_frm, image=viewer.img_logo, anchor=tk.W)
    logo.grid(row=0, column=0, sticky="nw")
    viewer.tk_logo = logo
    # row 1: viewer header text (center)
    lbl = tk.Label(header_frm, text="Viewer Header", font=("TkFixedFont 16"))
    lbl.grid(row=1, column=0, sticky="nsew")
    viewer.tk_header = lbl
    num_col, _ = header_frm.grid_size()  # config column sizes
    for i in range(num_col):
        header_frm.grid_columnconfigure(i, weight=1)


def create_footer(viewer) -> None:  # type: ignore
    """Create footer with controls and status bar

    Args:
        viewer (Viewer): PeekingDuck Viewer object
    """
    footer_frm = ttk.Frame(viewer.root, name="footer_frm")
    footer_frm.pack(side=tk.BOTTOM, fill=tk.X)
    viewer.tk_footer_frm = footer_frm
    # row 0: spacer
    lbl = tk.Label(footer_frm)  # row spacer
    lbl.grid(row=0, column=0)
    # row 1: controls
    ctrl_frm = ttk.Frame(footer_frm, name="ctrl_frm")
    ctrl_frm.grid(row=1, column=0, sticky="ew")
    _create_controls(viewer, ctrl_frm)
    # row 2: status bar
    lbl = tk.Label(footer_frm, anchor=tk.CENTER, text="Status bar text")
    lbl.grid(row=2, column=0, sticky="ew")
    viewer.tk_status_bar = lbl
    # row 3: spacer
    lbl = tk.Label(footer_frm)  # row spacer
    lbl.grid(row=3, column=0)

    num_col, _ = footer_frm.grid_size()  # config column sizes
    for i in range(num_col):
        footer_frm.grid_columnconfigure(i, weight=1)


def create_side_margins(viewer) -> None:  # type: ignore
    """Create left and right side margins

    Args:
        viewer (Viewer): PeekingDuck Viewer object
    """
    left_margin_frm = ttk.Frame(
        viewer.root, name="left_margin_frm", width=50, height=100
    )
    left_margin_frm.pack(side=tk.LEFT, fill=tk.NONE, expand=False)
    right_margin_frm = ttk.Frame(
        viewer.root, name="right_margin_frm", width=50, height=100
    )
    right_margin_frm.pack(side=tk.RIGHT, fill=tk.NONE, expand=False)


def create_body(viewer) -> None:  # type: ignore
    """Create body with video image and playlist in side panel

    Args:
        viewer (Viewer): PeekingDuck Viewer object
    """
    body_frm = ttk.Frame(viewer.root, name="body_frm")
    body_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.tk_body_frm = body_frm
    #
    # right side panel for playlist
    #
    panel_frm = ttk.Frame(body_frm, name="panel_frm")
    panel_frm.pack(side=tk.RIGHT, fill=tk.Y)
    viewer.tk_panel_frm = panel_frm
    # playlist top/bottom spacers:
    # need these for video image to center properly after hiding playlist
    lbl = tk.Label(panel_frm)
    lbl.pack(side=tk.TOP)
    lbl = tk.Label(panel_frm)
    lbl.pack(side=tk.BOTTOM)
    # playlist
    playlist_frm = ttk.Frame(panel_frm, name="playlist_frm")
    playlist_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.tk_playlist_frm = playlist_frm
    playlist_view = SingleColumnPlayListView(
        playlist=viewer.playlist, root=playlist_frm
    )
    viewer.tk_playlist_view = playlist_view
    playlist_view.register_callback("add", viewer.on_add_pipeline)
    playlist_view.register_callback("delete", viewer.on_delete_pipeline)
    playlist_view.register_callback("run", viewer.on_run_pipeline)
    viewer.playlist_show = True

    # video image
    image_frm = ttk.Frame(body_frm, name="image_frm")
    image_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.tk_image_frm = image_frm
    output_image = tk.Label(image_frm)
    output_image.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    viewer.tk_output_image = output_image


def _create_controls(viewer, ctrl_frm: ttk.Frame) -> None:  # type: ignore
    """Create 3 rows of controls: slider/progress bar/buttons

    Args:
        viewer (Viewer): PeekingDuck Viewer object
        body_frm (ttk.Frame): container frame for controls
    """
    #
    # row 0: info controls
    #
    # slider
    slider = ttk.Scale(
        ctrl_frm,
        orient=tk.HORIZONTAL,
        from_=1,
        to=100,
        command=viewer.sync_slider_to_frame,
    )
    slider.grid(row=0, column=0, columnspan=95, sticky="nsew")
    viewer.tk_scale = slider
    slider.grid_remove()  # hide it first
    # progress bar
    progress_bar = ttk.Progressbar(
        ctrl_frm,
        orient=tk.HORIZONTAL,
        length=100,
        mode="determinate",
        value=0,
        maximum=100,
    )
    progress_bar.grid(row=0, column=0, columnspan=95, sticky="nsew")
    viewer.tk_progress = progress_bar
    # frame number
    lbl = tk.Label(ctrl_frm, text="0", anchor=tk.W)
    lbl.grid(row=0, column=95, columnspan=BTN_WIDTH_SPAN, sticky="nsew")
    viewer.tk_lbl_frame_num = lbl
    #
    # row 1: buttons
    #
    btn_play = ttk.Button(ctrl_frm, text="Play", command=viewer.btn_play_stop_press)
    btn_play.grid(row=1, column=0, columnspan=BTN_WIDTH_SPAN, sticky="nsew")
    viewer.tk_btn_play = btn_play
    btn_zoom_out = ttk.Button(ctrl_frm, text="-", command=viewer.btn_zoom_out_press)
    btn_zoom_out.grid(row=1, column=92, columnspan=BTN_WIDTH_SPAN, sticky="nsew")
    viewer.tk_btn_zoom_out = btn_zoom_out
    lbl = tk.Label(ctrl_frm, text="100%")
    lbl.grid(row=1, column=93, columnspan=BTN_WIDTH_SPAN, sticky="nsew")
    viewer.tk_lbl_zoom = lbl
    btn_zoom_in = ttk.Button(ctrl_frm, text="+", command=viewer.btn_zoom_in_press)
    btn_zoom_in.grid(row=1, column=94, columnspan=BTN_WIDTH_SPAN, sticky="nsew")
    viewer.tk_btn_zoom_in = btn_zoom_in
    # spacer: without this, GUI will resize and flicker when frame number is updated
    lbl = tk.Label(ctrl_frm, text="          ")
    lbl.grid(row=1, column=95, columnspan=BTN_WIDTH_SPAN, sticky="nsew")
    #
    # row 2: playlist button
    #
    btn_playlist = ttk.Button(
        ctrl_frm, text="Playlist", command=viewer.btn_hide_show_playlist_press
    )
    btn_playlist.grid(row=2, column=94, columnspan=BTN_WIDTH_SPAN, sticky="nsew")
    viewer.tk_btn_playlist = btn_playlist
    # spacer: without this, GUI will resize and flicker when frame number is updated
    lbl = tk.Label(ctrl_frm, text="          ")
    lbl.grid(row=2, column=95, columnspan=BTN_WIDTH_SPAN, sticky="nsew")

    num_col, _ = ctrl_frm.grid_size()  # config column sizes
    for i in range(num_col):
        ctrl_frm.grid_columnconfigure(i, weight=1)
