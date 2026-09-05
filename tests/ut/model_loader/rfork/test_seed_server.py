# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import socket
import threading

import pytest

from vllm_ascend.model_loader.rfork.seed_server import (
    RForkSeedServerHandle,
    _create_bound_socket,
    start_rfork_server,
)


def test_bind_socket_uses_resolved_address_family(monkeypatch):
    calls = []

    def fake_getaddrinfo(*args):
        calls.append(args)
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    sock = _create_bound_socket("127.0.0.1")
    try:
        assert calls[0][2] == socket.AF_UNSPEC
        assert calls[0][3] == socket.SOCK_STREAM
        assert calls[0][5] == socket.AI_PASSIVE
        assert sock.family == socket.AF_INET
    finally:
        sock.close()


def test_seed_server_handle_stop_is_idempotent():
    class FakeServer:
        should_exit = False

    class FakeSocket:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    sock = FakeSocket()
    handle = RForkSeedServerHandle(server=FakeServer(), sock=sock, thread=thread, port=1234)
    assert handle.stop(timeout=0) is True
    assert handle.stop(timeout=0) is True
    assert handle.server.should_exit is True
    assert sock.closed == 1


def test_health_timeout_stops_and_joins_server(monkeypatch):
    class Response:
        status_code = 503

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.seed_server.requests.get",
        lambda *args, **kwargs: Response(),
    )
    result = start_rfork_server("key", ("session", {}, {}), health_timeout_sec=0.03, bind_host="127.0.0.1")
    assert result == -1
    assert not any(thread.name == "RForkSeedServer" and thread.is_alive() for thread in threading.enumerate())


@pytest.mark.skipif(not socket.has_ipv6, reason="IPv6 is not available")
def test_ipv6_bind_can_start_and_stop():
    handle = start_rfork_server(
        "key-ipv6",
        ("session", {}, {}),
        health_timeout_sec=2,
        bind_host="::1",
    )
    if handle == -1:
        pytest.skip("IPv6 loopback is not available on this host")
    assert isinstance(handle, RForkSeedServerHandle)
    assert handle.port > 0
    assert handle.stop() is True
