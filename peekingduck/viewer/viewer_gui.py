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

import tkinter as tk
from tkinter import ttk
from peekingduck.viewer.viewer_utils import load_image
from peekingduck.viewer.single_column_view import SingleColumnPlayListView

LOGO: str = "peekingduck/viewer/PeekingDuckLogo.png"
MIN_HEIGHT: int = 768
MIN_WIDTH: int = 1024
WIN_HEIGHT: int = 960
WIN_WIDTH: int = 1280
MAGNIFYING_GLASS_EMOJI = "\U0001F50D"
BLANK_EMOJI = "\u2800"


def create_window(viewer) -> None:  # type: ignore
    """Create PeekingDuck Viewer window with the following components:
        +---------------------------+
        | Logo       Name           |
        +---------------------------+
        |                           |
        |     Image     Playlist    |
        |                           |
        |          Controls         |
        |                           |
        +---------------------------+
        |         Status bar        |
        +---------------------------+

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
    # dotw technotes: Need to create footer before body to ensure footer controls
    #                 do not get covered when image is zoomed in
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
    lbl = tk.Label(header_frm)  # row spacer
    lbl.grid(row=0, column=0)
    viewer.img_logo = load_image(LOGO, resize_pct=0.10)
    logo = tk.Label(header_frm, image=viewer.img_logo)
    logo.grid(row=1, column=0, columnspan=2, sticky="nsew")
    viewer.tk_logo = logo
    lbl = tk.Label(
        header_frm, text="PeekingDuck Viewer Header", font=("arial 20")
    )
    lbl.grid(row=1, column=3, columnspan=4, sticky="nsew")
    viewer.tk_header = lbl
    lbl = tk.Label(header_frm)  # column spacer
    lbl.grid(row=1, column=9)
    lbl = tk.Label(header_frm)  # row spacer
    lbl.grid(row=2, column=0)
    num_col, _ = header_frm.grid_size()  # config column sizes
    for i in range(num_col):
        header_frm.grid_columnconfigure(i, weight=1)


def create_footer(viewer) -> None:  # type: ignore
    footer_frm = ttk.Frame(viewer.root, name="footer_frm")
    footer_frm.pack(side=tk.BOTTOM, fill=tk.X)
    viewer.tk_footer_frm = footer_frm
    lbl = tk.Label(footer_frm)  # row spacer
    lbl.grid(row=0, column=0)
    lbl = tk.Label(footer_frm)  # col spacer
    lbl.grid(row=1, column=0)
    lbl = tk.Label(footer_frm, anchor=tk.W, text="Status bar text")
    lbl.grid(row=1, column=2, columnspan=6, sticky="ew")
    viewer.tk_status_bar = lbl
    lbl = tk.Label(footer_frm)  # col spacer
    lbl.grid(row=1, column=9)
    lbl = tk.Label(footer_frm)  # row spacer
    lbl.grid(row=2, column=0)
    num_col, _ = footer_frm.grid_size()  # config column sizes
    for i in range(num_col):
        footer_frm.grid_columnconfigure(i, weight=1)


def create_body(viewer) -> None:  # type: ignore
    body_frm = ttk.Frame(viewer.root, name="body_frm")
    body_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.tk_body_frm = body_frm
    #
    # right side panel for playlist
    #
    panel_frm = ttk.Frame(body_frm, name="panel_frm")
    panel_frm.pack(side=tk.RIGHT, fill=tk.Y)
    viewer.tk_panel_frm = panel_frm
    lbl = tk.Label(panel_frm)  # panel col spacer
    lbl.pack(side=tk.BOTTOM)
    playlist_frm = ttk.Frame(
        panel_frm, name="playlist_frm", relief=tk.RIDGE, borderwidth=1
    )
    playlist_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.tk_playlist_frm = playlist_frm
    playlist_view = SingleColumnPlayListView(
        playlist=viewer.playlist, root=playlist_frm
    )
    viewer.tk_playlist_view = playlist_view
    playlist_view.register_callback("add", viewer.on_add_pipeline)
    playlist_view.register_callback("delete", viewer.on_delete_pipeline)
    playlist_view.register_callback("play", viewer.on_play_pipeline)
    viewer.playlist_show = True

    #####
    #
    # bottom frame for controls
    #
    #####
    ctrl_frm = ttk.Frame(body_frm, name="ctrl_frm")
    ctrl_frm.pack(side=tk.BOTTOM, fill=tk.X)
    #
    # row 1: info controls
    #
    # slider
    slider = ttk.Scale(
        ctrl_frm,
        orient=tk.HORIZONTAL,
        from_=1,
        to=100,
        command=viewer._sync_slider_to_frame,
    )
    slider.grid(row=0, column=2, columnspan=6, sticky="nsew")
    viewer.tk_scale = slider
    slider.bind("<Button-1>", viewer.slider_set_value)
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
    progress_bar.grid(row=0, column=2, columnspan=6, sticky="nsew")
    viewer.tk_progress = progress_bar
    # frame number
    lbl = tk.Label(ctrl_frm, text="0", anchor=tk.W)
    lbl.grid(row=0, column=8, sticky="nsew")
    viewer.tk_lbl_frame_num = lbl
    #
    # row 2: buttons
    #
    btn_play = ttk.Button(
        ctrl_frm, text="Play", command=viewer.btn_play_stop_press
    )
    btn_play.grid(row=1, column=2, sticky="nsew")
    viewer.tk_btn_play = btn_play
    btn_zoom_out = ttk.Button(
        ctrl_frm, text="-", command=viewer.btn_zoom_out_press
    )
    btn_zoom_out.grid(row=1, column=5, sticky="nsew")
    viewer.tk_btn_zoom_out = btn_zoom_out
    lbl = tk.Label(ctrl_frm, text=f"{MAGNIFYING_GLASS_EMOJI} 100%")
    lbl.grid(row=1, column=6, sticky="nsew")
    viewer.tk_lbl_zoom = lbl
    btn_zoom_in = ttk.Button(
        ctrl_frm, text="+", command=viewer.btn_zoom_in_press
    )
    btn_zoom_in.grid(row=1, column=7, sticky="nsew")
    viewer.tk_btn_zoom_in = btn_zoom_in
    lbl = tk.Label(
        ctrl_frm,
        # text=f"{BLANK_EMOJI} {BLANK_EMOJI} {BLANK_EMOJI} {BLANK_EMOJI}",
        text=f"{BLANK_EMOJI}",
    )  # spacer to stablise GUI flickering when resizing
    lbl.grid(row=1, column=8, sticky="nsew")
    #
    # row 3: playlist button
    #
    btn_playlist = ttk.Button(
        ctrl_frm, text="Playlist", command=viewer.btn_hide_show_playlist_press
    )
    btn_playlist.grid(row=2, column=7, sticky="nsew")
    viewer.tk_btn_playlist = btn_playlist
    lbl = tk.Label(
        ctrl_frm,
        # text=f"{BLANK_EMOJI} {BLANK_EMOJI} {BLANK_EMOJI} {BLANK_EMOJI}",
        text=f"{BLANK_EMOJI}",
    )  # spacer to stablise GUI flickering when resizing
    lbl.grid(row=2, column=8, sticky="nsew")

    num_col, _ = ctrl_frm.grid_size()  # config column sizes
    for i in range(num_col):
        ctrl_frm.grid_columnconfigure(i, weight=1)

    image_frm = ttk.Frame(body_frm, name="image_frm")
    image_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    viewer.tk_image_frm = image_frm
    output_image = tk.Label(image_frm)
    output_image.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    viewer.tk_output_image = output_image
