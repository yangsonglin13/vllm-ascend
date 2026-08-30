# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import json
import math
import threading
import time

import requests
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.types import RForkIdentity, SeedAdvertisement, SeedLease

REQUEST_TIMEOUT_SEC = 10.0
HEARTBEAT_LOG_EVERY_N = 4
RELEASE_MAX_RETRIES = 3
RELEASE_RETRY_BACKOFF_SEC = 0.1


def get_local_seed_key(
    tp_rank: int,
    model_url: str,
    model_deploy_strategy_name: str,
    compatibility_fingerprint: str,
    is_draft_worker: bool = False,
    pp_rank: int | None = None,
    ep_rank: int | None = None,
) -> str:
    if not model_url or not model_deploy_strategy_name:
        raise RuntimeError(
            f"RFork seed key is not set: model_url={model_url!r}, "
            f"model_deploy_strategy_name={model_deploy_strategy_name!r}. "
            "Configure both values through model_loader_extra_config or "
            "MODEL_URL and MODEL_DEPLOY_STRATEGY_NAME."
        )
    if not isinstance(compatibility_fingerprint, str) or not compatibility_fingerprint:
        raise RuntimeError(
            "RFork requires a compatibility fingerprint for the seed key; "
            "build one with _build_rfork_compatibility_fingerprint()."
        )

    descriptor = {
        "compatibility_fingerprint": str(compatibility_fingerprint),
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "ep_rank": ep_rank,
        "is_draft_worker": bool(is_draft_worker),
    }
    canonical_descriptor = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_descriptor.encode("utf-8")).hexdigest()


class RForkPlannerClient:
    def __init__(
        self,
        config: RForkConfig,
        identity: RForkIdentity,
        *,
        release_max_retries: int = RELEASE_MAX_RETRIES,
        release_retry_backoff_sec: float = RELEASE_RETRY_BACKOFF_SEC,
    ) -> None:
        request_timeout_sec = config.request_timeout_sec
        if isinstance(request_timeout_sec, bool) or not isinstance(request_timeout_sec, (int, float)):
            raise ValueError("request_timeout_sec must be a finite positive number")
        if not math.isfinite(float(request_timeout_sec)) or float(request_timeout_sec) <= 0:
            raise ValueError("request_timeout_sec must be a finite positive number")
        if (
            isinstance(release_max_retries, bool)
            or not isinstance(release_max_retries, int)
            or release_max_retries <= 0
        ):
            raise ValueError("release_max_retries must be a positive integer")
        if isinstance(release_retry_backoff_sec, bool) or not isinstance(release_retry_backoff_sec, (int, float)):
            raise ValueError("release_retry_backoff_sec must be a finite non-negative number")
        if not math.isfinite(float(release_retry_backoff_sec)) or float(release_retry_backoff_sec) < 0:
            raise ValueError("release_retry_backoff_sec must be a finite non-negative number")

        self.scheduler_url = config.scheduler_url
        self.tp_rank = identity.tp_rank
        self.request_timeout_sec = float(request_timeout_sec)
        self.release_max_retries = release_max_retries
        self.release_retry_backoff_sec = float(release_retry_backoff_sec)
        self.last_advertisement: SeedAdvertisement | None = None
        self.local_seed_key = get_local_seed_key(
            tp_rank=identity.tp_rank,
            model_url=config.model_url,
            model_deploy_strategy_name=config.model_deploy_strategy_name,
            compatibility_fingerprint=identity.compatibility_fingerprint,
            is_draft_worker=identity.is_draft_model,
            pp_rank=identity.pp_rank,
            ep_rank=identity.ep_rank,
        )

    def _require_scheduler(self) -> None:
        if not self.scheduler_url:
            raise RuntimeError(
                "rfork_scheduler_url is not set. Configure it through model_loader_extra_config or "
                "RFORK_SCHEDULER_URL."
            )

    def acquire_seed(self) -> SeedLease | None:
        try:
            self._require_scheduler()
            response = requests.get(
                f"{self.scheduler_url}/get_seed",
                headers={"SEED_KEY": self.local_seed_key},
                timeout=self.request_timeout_sec,
            )
            if response.status_code != 200:
                raise RuntimeError(f"planner get_seed returned status={response.status_code}")
            seed_ip = response.headers.get("SEED_IP")
            seed_port = response.headers.get("SEED_PORT")
            user_id = response.headers.get("USER_ID")
            seed_rank = response.headers.get("SEED_RANK")
            if not seed_ip or not seed_port or not user_id or seed_rank is None:
                raise RuntimeError("planner returned incomplete seed lease headers")
            parsed_port = int(seed_port)
            parsed_rank = int(seed_rank)
            if parsed_port <= 0 or parsed_rank < 0:
                raise ValueError
            return SeedLease(
                seed_ip=seed_ip,
                seed_port=parsed_port,
                user_id=user_id,
                seed_rank=parsed_rank,
                seed_key=self.local_seed_key,
            )
        except Exception as exc:
            logger.warning("RFork planner seed acquisition failed: %s", exc)
            return None

    def release_seed(self, lease: SeedLease) -> bool:
        try:
            self._require_scheduler()
        except RuntimeError as exc:
            logger.warning("RFork planner lease release setup failed: %s", exc)
            return False

        headers = {
            "SEED_IP": lease.seed_ip,
            "SEED_PORT": str(lease.seed_port),
            "USER_ID": lease.user_id,
            "SEED_RANK": str(lease.seed_rank),
        }
        for attempt in range(self.release_max_retries):
            try:
                response = requests.post(
                    f"{self.scheduler_url}/put_seed",
                    headers=headers,
                    timeout=self.request_timeout_sec,
                )
                if response.status_code in (200, 404):
                    return True
                logger.warning(
                    "RFork planner lease release attempt %d/%d returned status=%s",
                    attempt + 1,
                    self.release_max_retries,
                    response.status_code,
                )
            except Exception as exc:
                logger.warning(
                    "RFork planner lease release attempt %d/%d failed: %s",
                    attempt + 1,
                    self.release_max_retries,
                    exc,
                )
            if attempt + 1 < self.release_max_retries and self.release_retry_backoff_sec > 0:
                time.sleep(self.release_retry_backoff_sec * (attempt + 1))
        return False

    def remove_seed(self, advertisement: SeedAdvertisement | None = None) -> bool:
        try:
            self._require_scheduler()
            target = advertisement or self.last_advertisement
            if target is None:
                return True
            response = requests.post(
                f"{self.scheduler_url}/remove_seed",
                headers={
                    "SEED_KEY": self.local_seed_key,
                    "SEED_IP": target.seed_ip,
                    "SEED_PORT": str(target.seed_port),
                    "SEED_RANK": str(target.seed_rank),
                },
                timeout=self.request_timeout_sec,
            )
            if response.status_code not in (200, 404):
                logger.warning("RFork planner seed removal returned status=%s", response.status_code)
                return False
            if target == self.last_advertisement:
                self.last_advertisement = None
            return True
        except Exception as exc:
            logger.warning("RFork planner seed removal failed: %s", exc)
            return False

    def report_seed_once(self, port: int, seed_ip: str | None = None) -> bool:
        try:
            self._require_scheduler()
            advertisement = SeedAdvertisement(seed_ip or get_ip(), port, self.tp_rank)
            response = requests.post(
                f"{self.scheduler_url}/add_seed",
                headers={
                    "SEED_KEY": self.local_seed_key,
                    "SEED_IP": advertisement.seed_ip,
                    "SEED_PORT": str(advertisement.seed_port),
                    "SEED_RANK": str(advertisement.seed_rank),
                    "SEED_REFCNT": "0",
                },
                timeout=self.request_timeout_sec,
            )
            if response.status_code != 200:
                logger.warning("RFork planner seed report returned status=%s", response.status_code)
                return False
            self.last_advertisement = advertisement
            return True
        except Exception as exc:
            logger.warning("RFork planner seed report failed: %s", exc)
            return False

    def report_seed(
        self,
        port: int,
        sleep_interval: float = 30,
        stop_event: threading.Event | None = None,
        seed_ip: str | None = None,
        initial_delay: bool = False,
    ) -> None:
        if initial_delay and stop_event is not None and stop_event.wait(sleep_interval):
            return
        heartbeat_index = 0
        while stop_event is None or not stop_event.is_set():
            heartbeat_index += 1
            reported = self.report_seed_once(port, seed_ip=seed_ip)
            if not reported:
                logger.warning("RFork heartbeat failed for seed_key=%s", self.local_seed_key)
            elif heartbeat_index % HEARTBEAT_LOG_EVERY_N == 0:
                logger.debug("RFork heartbeat accepted for seed_key=%s", self.local_seed_key)
            if stop_event is None:
                time.sleep(sleep_interval)
            else:
                stop_event.wait(sleep_interval)
