# Copyright 2021 AI Singapore
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
Universal logging configuration
"""

import logging
import os
import sys
import traceback
from types import TracebackType
from typing import Dict, Optional, Type

from colorama import Fore, Style, init


class LoggerSetup:  # pylint: disable=too-few-public-methods
    """Set up the universal logging configuration"""

    def __init__(self, log_level: str = "info") -> None:
        if os.name == "nt":
            init()

        formatter = ColoredFormatter(
            "{asctime} {name} {level_color} {levelname}"
            "{reset}: {msg_color} {message} {reset}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
            colors={
                "DEBUG": Fore.RESET + Style.BRIGHT,
                "INFO": Fore.RESET + Style.BRIGHT,
                "WARNING": Fore.YELLOW + Style.BRIGHT,
                "ERROR": Fore.RED + Style.BRIGHT,
                "CRITICAL": Fore.RED + Style.BRIGHT,
            },
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger()
        self.logger.handlers[:] = []
        self.logger.addHandler(handler)
        sys.excepthook = self.handle_exception
        LoggerSetup.set_log_level(log_level)

    def handle_exception(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType,
    ) -> None:
        """Use Python's logging module when showing errors"""

        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
        traceback_msg = " ".join([str(elem) for elem in error_list[:-1]])
        error_msg = str(error_list[-1])

        # Make the error type more obvious in terminal by separating these
        self.logger.error(traceback_msg)
        self.logger.error(error_msg)

    @staticmethod
    def set_log_level(
        log_level: str = "info", logger_name: Optional[str] = None
    ) -> None:
        """Changes the logging level to the specified one.
        If logger_name is None, then this will change the global log level.

        Args:
            log_level (str, optional): Log level to set. Defaults to "info".
            logger_name (Optional[str], optional): Logger name. Defaults to None.
        """
        log_level_settings = set(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
        log_level = log_level.upper()
        if log_level not in log_level_settings:
            log_level = "INFO"
        logger = logging.getLogger(logger_name)
        if log_level != "INFO":
            if logger_name:
                logger.info(f"Changing log_level to {log_level}")
            else:
                logger.info(f"Changing global log_level to {log_level}")
        logger.setLevel(log_level)


class ColoredFormatter(logging.Formatter):
    """This class formats the color of logging messages"""

    def __init__(
        self, *args: str, colors: Optional[Dict[str, str]] = None, **kwargs: str
    ) -> None:
        """Initialize the formatter with specified format strings"""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as text."""

        record.level_color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL
        record.msg_color = self.colors.get(record.levelname, "")

        return super().format(record)
