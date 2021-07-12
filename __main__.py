# Copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from peekingduck.utils.logger import setup_logger
import peekingduck.runner as pkd

if __name__ == "__main__":
    RUN_PATH = os.path.join(os.getcwd(), 'PeekingDuck/run_config.yml')

    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Run path: %s", RUN_PATH)

    runner = pkd.Runner(RUN_PATH, None, "PeekingDuck")
    runner.run()
