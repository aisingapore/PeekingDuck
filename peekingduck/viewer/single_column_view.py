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

"""Implements the PlayList single-column list view class"""

from typing import Callable, Dict, Union
import logging
import tkinter as tk
from tkinter import ttk
from peekingduck.viewer.playlist import PipelineStats, PlayList

OP_LIST = ["add", "delete", "run"]  # Supported GUI operations
PLAYLIST_WIDTH = 200


class SingleColumnPlayListView:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """Use tk.ListBox as single-column list view"""

    def __init__(self, playlist: PlayList, root: tk.Widget):
        self.logger = logging.getLogger(__name__)
        self.playlist = playlist
        self.root = root
        self.view = None
        self._callback: Dict[str, Callable] = {}
        self._selected: Union[None, str] = None
        self._sort_desc = False
        self._tk_fg_system = ""  # different for diff platforms
        self.create_tk_widgets()
        self.redraw_view()

    def create_tk_widgets(self) -> None:
        """Create Tk widgets for the GUI view"""
        left_margin_frm = ttk.Frame(
            self.root, name="left_margin_frm", width=50, height=100
        )
        left_margin_frm.pack(side=tk.LEFT, fill=tk.NONE, expand=False)

        lbl = tk.Label(self.root, text="Pipelines:")
        lbl.pack(side=tk.TOP)
        lbl.bind("<Button-1>", self.header_press)
        self.header = lbl

        # playlist controls
        playlist_ctrl_frm = ttk.Frame(master=self.root, name="playlist_ctrl")
        playlist_ctrl_frm.pack(side=tk.BOTTOM)

        self._btn_add = ttk.Button(playlist_ctrl_frm, text="Add")
        self._btn_delete = ttk.Button(playlist_ctrl_frm, text="Delete")
        self._btn_play = ttk.Button(playlist_ctrl_frm, text="Run")
        # Tk technotes:
        # - if bind to Button (mouse down) event,
        #   the Play button will be "stuck" and show button down style (blue on macOS)
        #   when it is clicked on.
        # - binding to ButtonRelease (mouse up) event solves above issue
        self._btn_add.bind("<ButtonRelease-1>", self.btn_add_press)
        self._btn_delete.bind("<ButtonRelease-1>", self.btn_delete_press)
        self._btn_play.bind("<ButtonRelease-1>", self.btn_run_press)
        for child in playlist_ctrl_frm.winfo_children():
            child.pack(side=tk.LEFT)  # pack above buttons

        # info panel
        info_frm = ttk.Frame(master=self.root, name="playlist_info")
        info_frm.pack(side=tk.BOTTOM, fill=tk.X)

        lbl = tk.Label(info_frm, text="Pipeline Information:")
        # info labels
        lbl.grid(row=0, column=0, columnspan=2)
        lbl = tk.Label(info_frm, text="Name:", anchor=tk.E)
        lbl.grid(row=1, column=0, sticky="ne")
        lbl = tk.Label(info_frm, text="Modified:", anchor=tk.E)
        lbl.grid(row=2, column=0, sticky="ne")
        lbl = tk.Label(info_frm, text="Path:", anchor=tk.E)
        lbl.grid(row=3, column=0, sticky="ne")
        # playlist info
        self._info_name = tk.Message(
            info_frm, text="name", width=PLAYLIST_WIDTH, anchor=tk.W
        )
        self._info_name.grid(row=1, column=1, sticky="nw")
        self._info_datetime = tk.Message(
            info_frm, text="datetime", width=PLAYLIST_WIDTH, anchor=tk.W
        )
        self._info_datetime.grid(row=2, column=1, sticky="nw")
        self._info_path = tk.Message(
            info_frm, text="path", width=PLAYLIST_WIDTH, anchor=tk.W
        )
        self._info_path.grid(row=3, column=1, sticky="nw")

        num_col, _ = info_frm.grid_size()  # config column sizes
        for i in range(num_col):
            info_frm.grid_columnconfigure(i, weight=1)

        # listbox
        playlist_listbox = tk.Listbox(
            master=self.root, relief=tk.RIDGE, borderwidth=1, height=20
        )
        playlist_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        playlist_listbox.bind("<<ListboxSelect>>", self.selection_changed)
        self.tk_listbox = playlist_listbox

    def redraw_view(self) -> None:
        """Populate playlist contents"""
        self.sorted_playlist = sorted(self.playlist, reverse=self._sort_desc)
        self._pipeline_to_index_map: Dict[str, int] = {}
        self._index_to_stats_map: Dict[int, PipelineStats] = {}
        self.tk_listbox.delete(0, tk.END)
        for i, stats in enumerate(self.sorted_playlist):
            self._pipeline_to_index_map[stats.pipeline] = i
            self._index_to_stats_map[i] = stats
            self.tk_listbox.insert(i, stats.name)
            if len(self._tk_fg_system) == 0:
                # save platform specific foreground color
                self._tk_fg_system = self.tk_listbox["fg"]
                self.logger.debug(f"saving system foreground='{self._tk_fg_system}'")
            if len(stats.datetime) == 0:
                self.tk_listbox.itemconfig(i, {"fg": "red"})  # mark as error
        self.header["text"] = f"Pipelines: {'v' if self._sort_desc else '^'}"

    def get_selected_index(self) -> int:
        """Return index of selected listbox entry

        Returns:
            int: Index of selected entry
        """
        selection_indices = self.tk_listbox.curselection()
        i = selection_indices[0]  # only want first selection
        return i

    def register_callback(self, operation: str, player_callback: Callable) -> None:
        """Register callback function in Player to be called when playlist events are generated

        Args:
            operation (str): One of [ "add", "delete", "run" ]
            player_callback (Callable): The hook to the Player
        """
        if operation in OP_LIST:
            self._callback[operation] = player_callback
        else:
            raise ValueError(f"Unsupported callback operation: {operation}")

    def reset(self) -> None:
        """Force widget to recalculate width based on contents"""
        self.tk_listbox.config(width=0)

    def select(self, pipeline: str) -> None:
        """Select given pipeline on the GUI

        Args:
            pipeline (str): Pipeline to select.
        """
        self._selected = pipeline
        self.show_selected_pipeline()

    def show_selected_pipeline(self) -> None:
        """Show current selected pipeline by highlighting it in the listbox"""
        if self._selected:
            self.tk_listbox.selection_clear(0, tk.END)
            i = self._pipeline_to_index_map[self._selected]
            self.tk_listbox.select_set(i)
            self.update_pipeline_info(i)

    def update_pipeline_info(self, i: int) -> None:
        """Update pipeline info panel details"""
        stats = self._index_to_stats_map[i]
        self.logger.debug(f"show stats[{i}]: {stats.name}")
        self._info_name["text"] = stats.name
        if stats.datetime:
            display_datetime = f"{stats.datetime}"
            self._info_datetime["foreground"] = self._tk_fg_system
        else:
            display_datetime = "missing file"
            self._info_datetime["foreground"] = "red"
        self._info_datetime["text"] = display_datetime
        self._info_path["text"] = stats.pipeline

    #
    # Tk event callbacks
    #
    def btn_add_press(self, event: tk.Event) -> None:  # pylint: disable=unused-argument
        """Callback to handle "+" button

        Args:
            event (tk.Event): Tk event object
        """
        self.logger.debug("btn_add_press")
        if self._callback["add"]():
            self.redraw_view()

    def btn_delete_press(
        self, event: tk.Event  # pylint: disable=unused-argument
    ) -> None:
        """Callback to handle "-" button

        Args:
            event (tk.Event): Tk event object
        """
        i = self.get_selected_index()
        stats = self._index_to_stats_map[i]
        self.logger.debug(f"btn_delete_press: {stats.pipeline}")
        if self._callback["delete"](pipeline=stats.pipeline):
            self.redraw_view()

    def btn_run_press(self, event: tk.Event) -> None:  # pylint: disable=unused-argument
        """Callback to handle "Run" button

        Args:
            event (tk.Event): Tk event object
        """
        i = self.get_selected_index()
        stats = self._index_to_stats_map[i]
        self.logger.debug(f"btn_run_press: {stats.pipeline}")
        self._callback["run"](pipeline=stats.pipeline)

    def header_press(self, event: tk.Event) -> None:  # pylint: disable=unused-argument
        """Callback when header is clicked, changes sort order

        Args:
            event (tk.Event): Tk event object.
        """
        self._sort_desc = not self._sort_desc
        self.redraw_view()
        self.show_selected_pipeline()

    def selection_changed(
        self, event: tk.Event  # pylint: disable=unused-argument
    ) -> None:
        """Update pipeline info when user selects a pipeline in playlist

        Args:
            event (tk.Event): Tk event object
        """
        i = self.get_selected_index()
        self.update_pipeline_info(i)
