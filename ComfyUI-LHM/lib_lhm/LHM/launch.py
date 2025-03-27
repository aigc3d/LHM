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


import argparse
import pdb

from LHM.runners import REGISTRY_RUNNERS


def main():

    parser = argparse.ArgumentParser(description="OpenLRM launcher")
    parser.add_argument("runner", type=str, help="Runner to launch")
    args, unknown = parser.parse_known_args()

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError("Runner {} not found".format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]
    with RunnerClass() as runner:
        runner.run()


if __name__ == "__main__":
    main()
