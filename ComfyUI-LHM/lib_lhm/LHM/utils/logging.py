# Copyright (c) 2023-2024, Zexin He
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


import os
import logging
from tqdm.auto import tqdm


class TqdmStreamHandler(logging.StreamHandler):
    def emit(self, record):
        tqdm.write(self.format(record))


def configure_logger(stream_level, log_level, file_path = None):
    _stream_level = stream_level.upper()
    _log_level = log_level.upper()
    _project_level = _log_level

    _formatter = logging.Formatter("[%(asctime)s] %(name)s: [%(levelname)s] %(message)s")

    _stream_handler = TqdmStreamHandler()
    _stream_handler.setLevel(_stream_level)
    _stream_handler.setFormatter(_formatter)

    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        _file_handler = logging.FileHandler(file_path)
        _file_handler.setLevel(_log_level)
        _file_handler.setFormatter(_formatter)

    _project_logger = logging.getLogger(__name__.split('.')[0])
    _project_logger.setLevel(_project_level)
    _project_logger.addHandler(_stream_handler)
    if file_path is not None:
        _project_logger.addHandler(_file_handler)
