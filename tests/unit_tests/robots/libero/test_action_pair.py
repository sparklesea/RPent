# Copyright 2026 The RPent Authors.
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

"""Unit tests for generic open-loop and guarded action pairs."""

from types import SimpleNamespace

import pytest

from robots.libero.action_pair import execute_action_pair


class Driver:
    def __init__(self, pick_success=True, terminated=False):
        self.env = SimpleNamespace(terminated=terminated, truncated=False)
        self._last_obs_gripper = 0.08
        self._check_cancelled = lambda: None
        self.calls = []
        self.pick_success = pick_success

    def move_to(self, **kwargs):
        self.calls.append(("move_to", kwargs))
        return {"name": "move_to", "final_dist_m": 0.005}

    def pi0_pick(self, **kwargs):
        self.calls.append(("pi0_pick", kwargs))
        return {"name": "pick", "success": self.pick_success}

    def release(self, **kwargs):
        self.calls.append(("release", kwargs))
        return {"name": "release", "final_gripper_opening": 0.08}


FIRST_MOVE = {"action": "move_to", "args": {"xyz": [0.1, 0.0, 0.2], "gripper": -1}}
SECOND_PICK = {"action": "pi0_pick", "args": {"prompt": "grasp the object"}}


def test_open_loop_runs_second_without_first_success():
    d = Driver(pick_success=False)
    result = execute_action_pair(d, FIRST_MOVE, SECOND_PICK, mode="open_loop")
    assert [name for name, _ in d.calls] == ["move_to", "pi0_pick"]
    assert result["second_executed"]
    assert result["guard"]["mode"] == "open_loop"


def test_guarded_skips_second_when_pick_fails():
    first = {"action": "pi0_pick", "args": {"prompt": "grasp the object"}}
    second = {"action": "move_to", "args": {"xyz": [0.2, 0.0, 0.2], "gripper": 1}}
    d = Driver(pick_success=False)
    result = execute_action_pair(d, first, second, mode="guarded")
    assert [name for name, _ in d.calls] == ["pi0_pick"]
    assert not result["second_executed"]
    assert result["guard"]["action_success"] is False


def test_guarded_runs_second_when_move_reaches_target():
    d = Driver()
    result = execute_action_pair(d, FIRST_MOVE, SECOND_PICK, mode="guarded")
    assert len(d.calls) == 2
    assert result["second_executed"]
    assert result["guard"]["position_ok"]


def test_open_loop_stops_after_terminal_first():
    d = Driver(terminated=True)
    result = execute_action_pair(d, FIRST_MOVE, SECOND_PICK, mode="open_loop")
    assert not result["second_executed"]
    assert result["actions_executed"] == 0
    assert d.calls == []


@pytest.mark.parametrize(
    "second",
    [
        {"action": "pi0_pick", "args": {}},
        {"action": "move_to", "args": {"xyz": [0.1, 0.2, 0.3], "typo": True}},
        {"action": "move_to", "args": {"xyz": [0.1, 0.2]}},
    ],
)
def test_invalid_second_cannot_execute_first(second):
    driver = Driver()
    with pytest.raises(ValueError):
        execute_action_pair(driver, FIRST_MOVE, second, mode="open_loop")
    assert driver.calls == []
