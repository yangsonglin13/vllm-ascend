# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from examples.rfork.rfork_planner import Scheduler, Store


class _Clock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def _store(clock, *, ttl=60, lease_ttl=10, policy="fifo"):
    return Store(
        heartbeat_ttl_sec=ttl,
        lease_ttl_sec=lease_ttl,
        default_resource_points=1,
        scheduler=Scheduler(policy),
        time_fn=clock,
    )


def test_lease_gc_is_independent_from_heartbeat_gc():
    clock = _Clock()
    store = _store(clock, ttl=100, lease_ttl=5)
    store.add_seed(seed_key="key", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0)
    result = store.get_seed(seed_key="key")
    assert result is not None
    _, lease = result
    assert store.debug_snapshot()["lease_count"] == 1

    clock.value = 5
    assert store.gc_expired_leases() == 1
    assert store.debug_snapshot()["lease_count"] == 0
    # The seed remains live because its heartbeat TTL is independent.
    assert store.debug_snapshot()["seed_count"] == 1
    assert store.get_seed(seed_key="key") is not None
    assert lease.user_id not in store.debug_snapshot()["leases"]


def test_remove_seed_reclaims_active_leases_and_is_safe_for_missing_seed():
    clock = _Clock()
    store = _store(clock)
    store.add_seed(seed_key="key", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0)
    assert store.get_seed(seed_key="key") is not None
    assert store.remove_seed(seed_key="key", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0) is True
    assert store.debug_snapshot()["seed_count"] == 0
    assert store.debug_snapshot()["lease_count"] == 0
    assert store.remove_seed(seed_key="key", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0) is False


def test_lru_uses_last_allocation_not_last_heartbeat():
    clock = _Clock()
    store = _store(clock, policy="lru")
    store.add_seed(seed_key="key", seed_ip="127.0.0.1", seed_port=1234, seed_rank=0)
    clock.value = 1
    store.add_seed(seed_key="key", seed_ip="127.0.0.2", seed_port=1235, seed_rank=0)

    first = store.get_seed(seed_key="key")
    assert first is not None
    assert first[0].seed_ip == "127.0.0.1"
    assert store.put_seed(
        seed_ip=first[0].seed_ip,
        seed_port=first[0].seed_port,
        seed_rank=first[0].seed_rank,
        user_id=first[1].user_id,
    )

    # Heartbeats can advance independently without changing LRU ordering.
    clock.value = 2
    store.add_seed(seed_key="key", seed_ip="127.0.0.2", seed_port=1235, seed_rank=0)
    clock.value = 3
    second = store.get_seed(seed_key="key")
    assert second is not None
    assert second[0].seed_ip == "127.0.0.1"
