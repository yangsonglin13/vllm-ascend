# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading


def test_registered_block_snapshot_is_independent(tensor_runtime):
    backend = tensor_runtime.RForkTransferBackend()
    backend.registered_weight_blocks = [(100, 64)]
    snapshot = backend.snapshot_registered_weight_blocks()
    snapshot.append((200, 64))
    assert backend.registered_weight_blocks == [(100, 64)]
    backend.registered_weight_blocks.clear()
    assert snapshot == [(100, 64), (200, 64)]


def test_snapshot_waits_for_registration_state_update(tensor_runtime):
    backend = tensor_runtime.RForkTransferBackend()
    backend.registered_weight_blocks = [(100, 64)]
    entered, completed = threading.Event(), threading.Event()
    snapshots = []

    def read():
        entered.set()
        snapshots.append(backend.snapshot_registered_weight_blocks())
        completed.set()

    worker = threading.Thread(target=read, daemon=True)
    with backend._get_lifecycle_lock():
        worker.start()
        assert entered.wait(1)
        assert not completed.wait(0.05)
        backend._clear_registration_state()
    worker.join(1)
    assert not worker.is_alive()
    assert snapshots == [[]]
