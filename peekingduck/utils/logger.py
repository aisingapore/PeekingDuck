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
Universal logging configueration
"""

import sys
import logging
from typing import Optional, Dict

from colorama import init, Fore, Style


def setup_logger() -> None:
    """
    Universal logging configuration
    """

    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s %(name)s %(levelname)s:%(message)s',
    #                     datefmt="%Y-%m-%dT%H:%M:%S")

    # logger = logging.getLogger(__name__)

    init()

    formatter = ColoredFormatter(
        '{asctime}  {name} [{lineno}] {level_color} {levelname} {reset}: {msg_color} {message} {reset}',
        style='{', datefmt='%Y-%m-%d %H:%M:%S',
        colors={
            'DEBUG': Fore.CYAN + Style.BRIGHT,
            'INFO': Fore.GREEN + Style.BRIGHT,
            'WARNING': Fore.YELLOW + Style.BRIGHT,
            'ERROR': Fore.RED + Style.BRIGHT,
            'CRITICAL': Fore.RED + Style.BRIGHT,
        }
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Information!!!!")
    logger.debug("Debug!!!!")
    logger.warning("Warning!!!!")
    logger.error("Error!!!!")
    logger.critical("Critical!!!!")


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args: str, colors: Optional[Dict[str, str]] = None, **kwargs: str) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.level_color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL
        record.msg_color = Fore.BLUE + Style.BRIGHT

        return super().format(record)
