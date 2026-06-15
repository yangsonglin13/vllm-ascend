#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

from vllm.logger import logger


@dataclass
class ModelLoaderStageTiming:
    label: str
    default_load_weights_time: float = 0.0
    default_load_weights_calls: int = 0
    process_weights_after_loading_time: float = 0.0
    process_weights_after_loading_calls: int = 0


def _set_timed_attr(
    patches: list[tuple[object, str, object]],
    owner: object | None,
    attr_name: str,
    make_wrapper: Callable[[Any], Any],
) -> None:
    if owner is None or not hasattr(owner, attr_name):
        return

    original = getattr(owner, attr_name)
    setattr(owner, attr_name, make_wrapper(original))
    patches.append((owner, attr_name, original))


@contextmanager
def record_model_loader_stage_timing(label: str) -> Iterator[ModelLoaderStageTiming]:
    timing = ModelLoaderStageTiming(label=label)
    patches: list[tuple[object, str, object]] = []

    def make_load_weights_wrapper(original):
        @wraps(original)
        def timed_load_weights(*args, **kwargs):
            tic = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timing.default_load_weights_time += time.perf_counter() - tic
                timing.default_load_weights_calls += 1

        return timed_load_weights

    def make_process_weights_wrapper(original):
        @wraps(original)
        def timed_process_weights(*args, **kwargs):
            tic = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timing.process_weights_after_loading_time += time.perf_counter() - tic
                timing.process_weights_after_loading_calls += 1

        return timed_process_weights

    try:
        try:
            from vllm.model_executor.model_loader import default_loader
            from vllm.model_executor.model_loader import utils as loader_utils
        except Exception as e:
            logger.debug("Could not install model loader timing hooks for %s: %s", label, e)
        else:
            _set_timed_attr(
                patches,
                getattr(default_loader, "DefaultModelLoader", None),
                "load_weights",
                make_load_weights_wrapper,
            )
            _set_timed_attr(
                patches,
                loader_utils,
                "process_weights_after_loading",
                make_process_weights_wrapper,
            )
            _set_timed_attr(
                patches,
                default_loader,
                "process_weights_after_loading",
                make_process_weights_wrapper,
            )
        yield timing
    finally:
        for owner, attr_name, original in reversed(patches):
            setattr(owner, attr_name, original)
        logger.info(
            "model_loader stage details: label=%s, default_load_weights=%.4fs/%d, "
            "process_weights_after_loading=%.4fs/%d",
            timing.label,
            timing.default_load_weights_time,
            timing.default_load_weights_calls,
            timing.process_weights_after_loading_time,
            timing.process_weights_after_loading_calls,
        )
