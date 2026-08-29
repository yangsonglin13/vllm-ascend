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

import hashlib
import json
import math
import threading
import time
from urllib.error import HTTPError

import requests
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

REQUEST_TIMEOUT_SEC = 10.0
HEARTBEAT_LOG_EVERY_N = 4
RELEASE_MAX_RETRIES = 3
RELEASE_RETRY_BACKOFF_SEC = 0.1
# Keep the same header as the transfer backend.  The legacy spelling is
# accepted by servers/planners during rolling upgrades, but new requests use
# this canonical form.
AUTH_TOKEN_HEADER = "X-RFORK-TOKEN"
LEGACY_AUTH_TOKEN_HEADER = "X-RFork-Auth-Token"
RFORK_PROTOCOL_VERSION = 2
RFORK_SEED_KEY_VERSION = f"rfork-v{RFORK_PROTOCOL_VERSION}"


def get_local_seed_key(
    disaggregation_mode: str,
    node_rank: int,
    tp_rank: int,
    model_url: str,
    model_deploy_strategy_name: str,
    seed_key_separator: str = "$",
    is_draft_worker: bool = False,
    pp_rank: int | None = None,
    ep_rank: int | None = None,
    compatibility_fingerprint: str | None = None,
) -> str:
    if not model_url or not model_deploy_strategy_name:
        err_msg = (
            f"RFork seed key is not set: model_url={model_url!r}, "
            f"model_deploy_strategy_name={model_deploy_strategy_name!r}. "
            "Ensure model_loader_extra_config contains "
            "`model_url` and `model_deploy_strategy_name`, or set "
            "MODEL_URL and MODEL_DEPLOY_STRATEGY_NAME."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Keep the pre-v2 form when no fingerprint is supplied.  This permits a
    # rolling upgrade with older planners and preserves compatibility for
    # callers which do not yet know the complete model descriptor.  The
    # production loader always supplies a fingerprint and therefore gets the
    # opaque, collision-free v2 form below.
    if compatibility_fingerprint is None:
        seed_key = f"{model_url}{seed_key_separator}{model_deploy_strategy_name}"
        key_parts = [disaggregation_mode, str(node_rank)]
        if pp_rank is not None:
            key_parts.append(f"pp{pp_rank}")
        key_parts.append(str(tp_rank))
        if ep_rank is not None:
            key_parts.append(f"ep{ep_rank}")
        if is_draft_worker:
            key_parts.append("draft")
        return f"{seed_key}{seed_key_separator}{seed_key_separator.join(key_parts)}"

    descriptor = {
        "version": RFORK_SEED_KEY_VERSION,
        "compatibility_fingerprint": str(compatibility_fingerprint),
        "model_url": model_url,
        "model_deploy_strategy_name": model_deploy_strategy_name,
        "disaggregation_mode": disaggregation_mode,
        "node_rank": node_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "ep_rank": ep_rank,
        "is_draft_worker": bool(is_draft_worker),
    }
    canonical_descriptor = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical_descriptor.encode("utf-8")).hexdigest()
    return f"{RFORK_SEED_KEY_VERSION}:{digest}"


class RForkSeedProtocol:
    def __init__(
        self,
        *,
        disaggregation_mode: str,
        node_rank: int,
        tp_rank: int,
        scheduler_url: str,
        model_url: str,
        model_deploy_strategy_name: str,
        seed_key_separator: str = "$",
        is_draft_worker: bool = False,
        pp_rank: int | None = None,
        ep_rank: int | None = None,
        compatibility_fingerprint: str | None = None,
        request_timeout_sec: float = REQUEST_TIMEOUT_SEC,
        auth_token: str | None = None,
        release_max_retries: int = RELEASE_MAX_RETRIES,
        release_retry_backoff_sec: float = RELEASE_RETRY_BACKOFF_SEC,
    ):
        self.disaggregation_mode = disaggregation_mode
        self.node_rank = node_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.ep_rank = ep_rank
        self.scheduler_url = scheduler_url
        self.model_url = model_url
        self.model_deploy_strategy_name = model_deploy_strategy_name
        self.seed_key_separator = seed_key_separator
        self.is_draft_worker = is_draft_worker
        self.compatibility_fingerprint = compatibility_fingerprint
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
        self._request_timeout = float(request_timeout_sec)
        self.auth_token = auth_token if isinstance(auth_token, str) and auth_token else None
        self.release_max_retries = release_max_retries
        self.release_retry_backoff_sec = float(release_retry_backoff_sec)
        self._last_report: dict[str, object] | None = None

        self._local_seed_key = get_local_seed_key(
            disaggregation_mode=self.disaggregation_mode,
            node_rank=self.node_rank,
            tp_rank=self.tp_rank,
            model_url=self.model_url,
            model_deploy_strategy_name=self.model_deploy_strategy_name,
            seed_key_separator=self.seed_key_separator,
            is_draft_worker=self.is_draft_worker,
            pp_rank=self.pp_rank,
            ep_rank=self.ep_rank,
            compatibility_fingerprint=self.compatibility_fingerprint,
        )

    def get_local_seed_key(self) -> str:
        return self._local_seed_key

    def _request_timeout_sec(self) -> float:
        return self._request_timeout

    def _auth_headers(self) -> dict[str, str]:
        if self.auth_token is None:
            return {}
        return {AUTH_TOKEN_HEADER: self.auth_token}

    def _ensure_scheduler_url_set(self) -> None:
        if not self.scheduler_url:
            raise RuntimeError(
                "rfork_scheduler_url is not set. Set it through model_loader_extra_config or "
                "VLLM_ASCEND_RFORK_SCHEDULER_URL."
            )

    def get_seed(self):
        try:
            self._ensure_scheduler_url_set()
            seed_key = self.get_local_seed_key()
            response = requests.get(
                f"{self.scheduler_url}/get_seed",
                headers={
                    "SEED_KEY": seed_key,
                    **self._auth_headers(),
                },
                timeout=self._request_timeout_sec(),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get seed from the planner, {response.status_code}, seed_key={seed_key!r}"
                )

            seed_ip = response.headers.get("SEED_IP")
            seed_port = response.headers.get("SEED_PORT")
            user_id = response.headers.get("USER_ID")
            seed_rank = response.headers.get("SEED_RANK")
            if not seed_ip or not seed_port or not user_id or not seed_rank:
                raise RuntimeError("Planner returned incomplete seed lease headers")
            try:
                if int(seed_port) <= 0 or int(seed_rank) < 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Planner returned invalid seed lease headers") from exc
            logger.debug(
                "seed_ip: %s, seed_port: %s, user_id: %s, seed_rank: %s",
                seed_ip,
                seed_port,
                user_id,
                seed_rank,
            )
            return {
                "seed_ip": seed_ip,
                "seed_port": seed_port,
                "user_id": user_id,
                "seed_rank": seed_rank,
                "seed_key": seed_key,
            }

        except RuntimeError as e:
            logger.warning("get_seed from scheduler RuntimeError: %s", e)
            return None
        except HTTPError as e:
            logger.exception("get_seed from scheduler HTTPError: %s", e)
            return None
        except Exception as e:
            logger.exception("get_seed from scheduler Exception: %s", e)
            return None

    def release_seed(self, seed) -> bool:
        if not isinstance(seed, dict):
            return False

        try:
            self._ensure_scheduler_url_set()
            user_id = seed["user_id"]
            seed_ip = seed["seed_ip"]
            seed_port = str(seed["seed_port"])
            seed_rank = str(seed["seed_rank"])
        except (RuntimeError, KeyError, TypeError) as e:
            logger.warning("release_seed input/setup failed: %s", e)
            return False

        headers = {
            "SEED_IP": seed_ip,
            "SEED_PORT": seed_port,
            "USER_ID": user_id,
            "SEED_RANK": seed_rank,
            **self._auth_headers(),
        }
        for attempt in range(self.release_max_retries):
            try:
                response = requests.post(
                    f"{self.scheduler_url}/put_seed",
                    headers=headers,
                    timeout=self._request_timeout_sec(),
                )
                if response.status_code == 200:
                    return True
                # Lease release is idempotent. A 404 commonly means the
                # planner's lease TTL already reclaimed the lease, which is
                # the desired final state and must not pin the worker forever.
                if response.status_code == 404:
                    logger.info("release_seed lease was already reclaimed by the planner.")
                    return True
                logger.warning(
                    "release_seed attempt %d/%d returned status=%s",
                    attempt + 1,
                    self.release_max_retries,
                    response.status_code,
                )
            except Exception as e:
                # requests raises several concrete exception types depending
                # on the transport; all are bounded by the request timeout.
                logger.warning(
                    "release_seed attempt %d/%d failed: %s",
                    attempt + 1,
                    self.release_max_retries,
                    e,
                )
            if attempt + 1 < self.release_max_retries and self.release_retry_backoff_sec > 0:
                time.sleep(self.release_retry_backoff_sec * (attempt + 1))
        return False

    def remove_seed(
        self,
        seed: dict[str, object] | None = None,
        *,
        port: int | None = None,
        seed_ip: str | None = None,
        seed_rank: int | None = None,
    ) -> bool:
        """Best-effort removal of this worker's advertised seed."""
        try:
            self._ensure_scheduler_url_set()
            report = self._last_report or {}
            if isinstance(seed, dict):
                port = port if port is not None else seed.get("seed_port")  # type: ignore[assignment]
                seed_ip = seed_ip or seed.get("seed_ip")  # type: ignore[assignment]
                seed_rank = seed_rank if seed_rank is not None else seed.get("seed_rank")  # type: ignore[assignment]
            reported_port = port if port is not None else report.get("seed_port")
            reported_ip = seed_ip or report.get("seed_ip") or get_ip()
            reported_rank = seed_rank if seed_rank is not None else report.get("seed_rank", self.tp_rank)
            if reported_port is None:
                return False
            headers = {
                "SEED_KEY": self.get_local_seed_key(),
                "SEED_IP": str(reported_ip),
                "SEED_PORT": str(reported_port),
                "SEED_RANK": str(reported_rank),
                **self._auth_headers(),
            }
            response = requests.post(
                f"{self.scheduler_url}/remove_seed",
                headers=headers,
                timeout=self._request_timeout_sec(),
            )
            # Removal is idempotent: a planner that already GC'd the seed is
            # still in the desired state.
            if response.status_code not in (200, 404):
                logger.warning("remove_seed returned status=%s", response.status_code)
                return False
            self._last_report = None
            return True
        except Exception as e:
            logger.warning("remove_seed best-effort cleanup failed: %s", e)
            return False

    def report_seed(
        self,
        port: int,
        sleep_interval: float = 30,
        stop_event: threading.Event | None = None,
        seed_ip: str | None = None,
        initial_delay: bool = False,
    ):
        heartbeat_idx = 0
        log_every_n = HEARTBEAT_LOG_EVERY_N
        if initial_delay:
            if stop_event is not None:
                if stop_event.wait(sleep_interval):
                    return
            else:
                time.sleep(sleep_interval)
        while stop_event is None or not stop_event.is_set():
            heartbeat_idx += 1
            result = self.report_seed_once(port, seed_ip=seed_ip)
            seed_key = self.get_local_seed_key()

            # Keep heartbeat frequency unchanged, but reduce log noise.
            # Always print failures immediately; keep success in debug logs.
            if result:
                if heartbeat_idx % log_every_n == 0:
                    logger.debug(
                        "[rfork_heartbeat] report seed to planner result: %s (%d/%d), seed_key=%s",
                        result,
                        heartbeat_idx % log_every_n if heartbeat_idx % log_every_n != 0 else log_every_n,
                        log_every_n,
                        seed_key,
                    )
            else:
                logger.warning(
                    "[rfork_heartbeat] report seed to planner result: %s (%d/%d), seed_key=%s",
                    result,
                    heartbeat_idx % log_every_n if heartbeat_idx % log_every_n != 0 else log_every_n,
                    log_every_n,
                    seed_key,
                )
            if stop_event is not None:
                stop_event.wait(sleep_interval)
            else:
                time.sleep(sleep_interval)

    def report_seed_once(self, port: int, seed_ip: str | None = None) -> bool:
        """Advertise one healthy seed, returning whether the planner accepted it."""

        try:
            self._ensure_scheduler_url_set()
            advertised_ip = seed_ip or get_ip()
            seed_key = self.get_local_seed_key()
            logger.debug("[rfork_heartbeat] reporting seed key: %s", seed_key)
            response = requests.post(
                f"{self.scheduler_url}/add_seed",
                headers={
                    "SEED_KEY": seed_key,
                    "SEED_IP": advertised_ip,
                    "SEED_PORT": str(port),
                    "SEED_RANK": str(self.tp_rank),
                    "SEED_REFCNT": str(0),
                    **self._auth_headers(),
                },
                timeout=self._request_timeout_sec(),
            )
            if response.status_code != 200:
                logger.warning("report_seed to planner returned status=%s", response.status_code)
                return False
            self._last_report = {
                "seed_ip": advertised_ip,
                "seed_port": port,
                "seed_rank": self.tp_rank,
            }
            return True
        except HTTPError as e:
            logger.warning("report_seed to planner HTTPError: %s", e)
        except Exception as e:
            logger.warning("report_seed to planner Exception: %s", e)
        return False
