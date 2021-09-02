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

import logging

import coloredlogs

LEVEL_COLOR = {'info': {'bold': True, 'color': 'green'},
               'warning': {'bold': True, 'color': 'yellow'},
               'error': {'bold': True, 'color': 'red'},
               'critical': {'bold': True, 'color': 'red'}}

DEFAULT_FIELD_STYLES = {'asctime': {'color': 'green'},
                        'hostname': {'color': 'yellow'},
                        'levelname': {'color': 'blue', 'bold': True},
                        'name': {'color': 'magenta'}}

def setup_logger():
    """
    Universal logging configuration
    """

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                        datefmt="%Y-%m-%dT%H:%M:%S")

    logger = logging.getLogger(__name__)

    coloredlogs.install(level_styles = LEVEL_COLOR,
                        field_styles = DEFAULT_FIELD_STYLES)

    logger.info("Information!!!!")
    logger.debug("Debug!!!!")
    logger.warning("Warning!!!!")
    logger.error("Error!!!!")
    logger.critical("Critical!!!!")

