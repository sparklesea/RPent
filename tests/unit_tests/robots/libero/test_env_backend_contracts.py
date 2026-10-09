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

"""RLinf package-layout compatibility without importing a simulator."""

from types import SimpleNamespace

import pytest

from robots.libero import env_server


@pytest.mark.parametrize("legacy", [False, True])
def test_load_backend_uses_available_namespace(monkeypatch, legacy):
    calls = []
    backend = type("Backend", (), {})
    benchmark = object()

    def load(name):
        calls.append(name)
        if legacy and name.startswith("rlinf.envs.sim"):
            raise ModuleNotFoundError(
                "No module named rlinf.envs.sim", name="rlinf.envs.sim"
            )
        if name.endswith(".libero_env"):
            return SimpleNamespace(LiberoEnv=backend)
        return SimpleNamespace(benchmark=benchmark)

    monkeypatch.setattr(env_server.importlib, "import_module", load)
    assert env_server._load_libero_backend() == (backend, benchmark)
    expected = "rlinf.envs.libero" if legacy else "rlinf.envs.sim.libero"
    assert calls[-1] == expected + ".utils"


def test_missing_transitive_dependency_does_not_fall_back(monkeypatch):
    calls = []

    def load(name):
        calls.append(name)
        raise ModuleNotFoundError("No module named dependency", name="dependency")

    monkeypatch.setattr(env_server.importlib, "import_module", load)
    with pytest.raises(ModuleNotFoundError, match="dependency"):
        env_server._load_libero_backend()
    assert calls == ["rlinf.envs.sim.libero.libero_env"]
