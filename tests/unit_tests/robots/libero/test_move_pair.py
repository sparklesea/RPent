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

"""Guard behavior for the opt-in pair executor, without a simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.move_pair import execute_move_pair


class Driver:
    def __init__(self, error=0, terminated=False, truncated=False, gap_change=0):
        self.env = SimpleNamespace(terminated=False, truncated=False)
        self._last_obs_eef_pos = np.array([0.0, 0.0, 0.2])
        self._last_obs_gripper = 0.04
        self.calls = []
        self.error, self.done, self.truncated, self.gap_change = (
            error,
            terminated,
            truncated,
            gap_change,
        )

    def _check_cancelled(self):
        pass

    def move_to(self, **kwargs):
        self.calls.append(kwargs)
        self._last_obs_eef_pos = np.array(kwargs["xyz"]) + [self.error, 0.0, 0.0]
        self._last_obs_gripper += self.gap_change
        self.env.terminated = self.done
        self.env.truncated = self.truncated
        return {"final_dist_m": self.error}


FIRST = {"xyz": [0.1, 0.0, 0.2], "gripper": 1}
SECOND = {"xyz": [0.2, 0.0, 0.2], "gripper": 1}


def test_second_runs_once_first_state_matches():
    d = Driver()
    r = execute_move_pair(d, FIRST, SECOND)
    assert len(d.calls) == 2 and r["second_executed"]
    assert r["guard"]["position_ok"] and r["guard"]["gripper_ok"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": 0.04},
        {"terminated": True},
        {"truncated": True},
        {"gap_change": 0.02},
        {"error": float("nan")},
    ],
)
def test_guard_rejects_second_motion(kwargs):
    d = Driver(**kwargs)
    r = execute_move_pair(d, FIRST, SECOND)
    assert len(d.calls) == 1 and not r["second_executed"]


@pytest.mark.parametrize(
    "second",
    [
        {"xyz": [0.2, 0.0, 0.2], "gripper": -1},
        {"xyz": [float("nan"), 0.0, 0.2], "gripper": 1},
        {"xyz": [0.2, 0.0, 0.2], "gripper": 1, "type": "pi0_pick"},
        {"xyz": [0.2, 0.0, 0.2], "gripper": 1, "max_steps": -1},
        {"xyz": [0.5, 0.0, 0.2], "gripper": 1},
    ],
)
def test_invalid_second_is_rejected_before_first_executes(second):
    d = Driver()
    with pytest.raises(ValueError):
        execute_move_pair(d, FIRST, second)
    assert d.calls == []


def test_episode_already_done_does_not_move():
    d = Driver()
    d.env.terminated = True
    r = execute_move_pair(d, FIRST, SECOND)
    assert r["actions_executed"] == 0
    assert not d.calls
