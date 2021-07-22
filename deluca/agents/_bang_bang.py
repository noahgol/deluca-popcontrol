# Copyright 2021 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from deluca.core import Agent
from deluca.core import field


class BangBang(Agent):
    target: float = field(0.0, jaxed=False)
    min_action: float = field(0.0, jaxed=False)
    max_action: float = field(100.0, jaxed=False)

    def init(self):
        return None

    def __call__(self, state, obs):
        """Assume observation is env state"""
        return state, self.min_action if obs > self.target else self.max_action
