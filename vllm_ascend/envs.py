#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


def _get_env_with_legacy_fallback(
    canonical_name: str,
    legacy_name: str | None = None,
    default: str | None = None,
) -> str | None:
    """Read a canonical variable and, for compatibility, its legacy alias.

    RFork is intentionally the only caller of the legacy aliases.  Keeping
    the lookup here prevents old deployment manifests from leaking raw
    ``os.environ`` access into the loader, while giving the canonical names
    well-defined precedence.
    """

    value = os.getenv(canonical_name)
    if value is None and legacy_name is not None:
        value = os.getenv(legacy_name)
    return default if value is None else value


env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # RFork logical model identity. The default is None (unset). The legacy
    # MODEL_URL variable is accepted only as a compatibility fallback.
    "VLLM_ASCEND_RFORK_MODEL_URL": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_MODEL_URL", "MODEL_URL", None
    ),
    # RFork deployment strategy identity. The default is None (unset). The
    # legacy MODEL_DEPLOY_STRATEGY_NAME variable remains a fallback.
    "VLLM_ASCEND_RFORK_DEPLOY_STRATEGY_NAME": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_DEPLOY_STRATEGY_NAME", "MODEL_DEPLOY_STRATEGY_NAME", None
    ),
    # RFork planner URL. The default is None (unset). RFORK_SCHEDULER_URL is
    # retained as a legacy fallback.
    "VLLM_ASCEND_RFORK_SCHEDULER_URL": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_SCHEDULER_URL", "RFORK_SCHEDULER_URL", None
    ),
    # RFork seed HTTP health timeout in seconds. Default: 5.0; valid range is
    # finite values greater than zero. RFORK_SEED_TIMEOUT_SEC is a fallback.
    "VLLM_ASCEND_RFORK_SEED_TIMEOUT_SEC": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_SEED_TIMEOUT_SEC", "RFORK_SEED_TIMEOUT_SEC", "5.0"
    ),
    # RFork planner/seed HTTP request timeout in seconds. Default: 10.0; valid
    # range is finite values greater than zero. RFORK_REQUEST_TIMEOUT_SEC is a
    # fallback.
    "VLLM_ASCEND_RFORK_REQUEST_TIMEOUT_SEC": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_REQUEST_TIMEOUT_SEC", "RFORK_REQUEST_TIMEOUT_SEC", "10.0"
    ),
    # RFork seed-key separator. Default: "$"; any non-empty string is valid.
    # RFORK_SEED_KEY_SEPARATOR is a fallback.
    "VLLM_ASCEND_RFORK_SEED_KEY_SEPARATOR": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_SEED_KEY_SEPARATOR", "RFORK_SEED_KEY_SEPARATOR", "$"
    ),
    # Optional shared RFork authentication token. Default: None (disabled).
    # This value is sensitive and must never be logged. RFORK_AUTH_TOKEN is a
    # fallback for existing deployments.
    "VLLM_ASCEND_RFORK_AUTH_TOKEN": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_AUTH_TOKEN", "RFORK_AUTH_TOKEN", None
    ),
    # Local host/interface for the RFork seed HTTP server. Default:
    # 0.0.0.0; any valid bind address is accepted. RFORK_SEED_BIND_HOST is a
    # fallback.
    "VLLM_ASCEND_RFORK_SEED_BIND_HOST": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_SEED_BIND_HOST", "RFORK_SEED_BIND_HOST", "0.0.0.0"
    ),
    # Advertised host/interface for planner seed records. Default: None,
    # meaning derive the local address. RFORK_SEED_ADVERTISE_HOST is a
    # fallback.
    "VLLM_ASCEND_RFORK_SEED_ADVERTISE_HOST": lambda: _get_env_with_legacy_fallback(
        "VLLM_ASCEND_RFORK_SEED_ADVERTISE_HOST", "RFORK_SEED_ADVERTISE_HOST", None
    ),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
