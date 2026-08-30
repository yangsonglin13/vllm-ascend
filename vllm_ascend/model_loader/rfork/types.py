# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class RForkLifecycleState(Enum):
    INITIALIZED = auto()
    LEASED = auto()
    REGISTERED = auto()
    SERVING = auto()
    FINALIZED = auto()


@dataclass(frozen=True, slots=True)
class RForkIdentity:
    disaggregation_mode: str
    node_rank: int
    tp_rank: int
    device_id: int
    is_draft_model: bool = False
    pp_rank: int | None = None
    ep_rank: int | None = None
    compatibility_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class SeedLease:
    seed_ip: str
    seed_port: int
    user_id: str
    seed_rank: int
    seed_key: str


@dataclass(frozen=True, slots=True)
class SeedAdvertisement:
    seed_ip: str
    seed_port: int
    seed_rank: int


@dataclass(frozen=True, slots=True)
class SeedTransferInfo:
    session_id: str
    weights: Mapping[str, Any]
    shapes: Mapping[str, Any] | None = None
