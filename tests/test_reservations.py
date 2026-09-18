"""会议预约接口测试：半开区间冲突、幂等重放、事务原子性与并发。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import reservations as core
from app.main import app


@pytest.fixture()
def db_path(tmp_path, monkeypatch) -> str:
    """每个测试用独立的 SQLite 文件。"""
    path = str(tmp_path / "reservations.db")
    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    return path


@pytest.fixture()
def client(db_path) -> TestClient:
    return TestClient(app)


def _payload(**overrides):
    payload = {
        "request_id": "req-1",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice", "bob"],
    }
    payload.update(overrides)
    return payload


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


# ------------------------------------------------------------------ 基本创建

def test_create_returns_201_and_normalizes_participants(client):
    resp = client.post("/api/v1/reservations",
                       json=_payload(participant_ids=["bob", "alice", "bob"]))
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["reservation_id"] == 1
    assert data["request_id"] == "req-1"
    assert data["start_utc"] == "2026-03-09T13:00:00Z"
    assert data["end_utc"] == "2026-03-09T14:00:00Z"
    assert data["participants"] == ["alice", "bob"]  # 去重 + 字典序
    assert data["created_at"]


def test_end_must_be_later_than_start(client):
    # 相等
    resp = client.post("/api/v1/reservations",
                       json=_payload(end_utc="2026-03-09T13:00:00Z"))
    assert resp.status_code == 422
    # 倒置
    resp = client.post("/api/v1/reservations",
                       json=_payload(end_utc="2026-03-09T12:00:00Z"))
    assert resp.status_code == 422


def test_naive_timestamp_rejected(client):
    resp = client.post("/api/v1/reservations",
                       json=_payload(start_utc="2026-03-09T13:00:00"))
    assert resp.status_code == 422


def test_empty_participants_rejected(client):
    resp = client.post("/api/v1/reservations", json=_payload(participant_ids=[]))
    assert resp.status_code == 422


# ------------------------------------------------------------------ 冲突检测

def test_adjacent_intervals_do_not_conflict(client):
    assert client.post("/api/v1/reservations", json=_payload()).status_code == 201
    # 紧邻其后（半开区间，end == start 不冲突）
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-2",
        start_utc="2026-03-09T14:00:00Z",
        end_utc="2026-03-09T15:00:00Z",
    ))
    assert resp.status_code == 201, resp.text
    # 紧邻其前
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-3",
        start_utc="2026-03-09T12:00:00Z",
        end_utc="2026-03-09T13:00:00Z",
    ))
    assert resp.status_code == 201, resp.text


@pytest.mark.parametrize("start,end", [
    ("2026-03-09T13:30:00Z", "2026-03-09T13:45:00Z"),  # 被包含
    ("2026-03-09T12:00:00Z", "2026-03-09T15:00:00Z"),  # 包含已有
    ("2026-03-09T12:30:00Z", "2026-03-09T13:30:00Z"),  # 左侧重叠
    ("2026-03-09T13:30:00Z", "2026-03-09T14:30:00Z"),  # 右侧重叠
    ("2026-03-09T13:00:00Z", "2026-03-09T14:00:00Z"),  # 完全相同
])
def test_any_half_open_overlap_conflicts(client, start, end):
    assert client.post("/api/v1/reservations", json=_payload()).status_code == 201
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-2", start_utc=start, end_utc=end,
        participant_ids=["alice"],
    ))
    assert resp.status_code == 409


def test_conflict_409_reports_reservation_and_participant(client):
    first = client.post("/api/v1/reservations", json=_payload()).json()
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-2",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T14:30:00Z",
        participant_ids=["bob", "carol"],
    ))
    assert resp.status_code == 409
    data = resp.json()
    assert data["conflicts"] == [
        {"reservation_id": first["reservation_id"], "participant_id": "bob"}
    ]


def test_conflict_leaves_no_partial_writes(client):
    assert client.post("/api/v1/reservations", json=_payload()).status_code == 201
    # alice 冲突 → carol 也不能被部分写入
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-2",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T14:30:00Z",
        participant_ids=["alice", "carol"],
    ))
    assert resp.status_code == 409
    # carol 在同一时段应能立即预约成功（证明没有残留占用）
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-3",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T14:30:00Z",
        participant_ids=["carol"],
    ))
    assert resp.status_code == 201, resp.text


def test_disjoint_participants_same_slot_ok(client):
    assert client.post("/api/v1/reservations", json=_payload()).status_code == 201
    resp = client.post("/api/v1/reservations",
                       json=_payload(request_id="req-2",
                                     participant_ids=["carol", "dave"]))
    assert resp.status_code == 201, resp.text


# ---------------------------------------------------------------------- 幂等

def test_idempotent_replay_returns_original_without_new_record(client):
    first = client.post("/api/v1/reservations",
                        json=_payload(participant_ids=["bob", "alice"]))
    assert first.status_code == 201
    # 同一 request_id + 同一规范化内容（参与者顺序/重复不同也算同一请求）
    replay = client.post("/api/v1/reservations",
                         json=_payload(participant_ids=["alice", "bob", "alice"]))
    assert replay.status_code == 200
    assert replay.json() == first.json()
    # 不新增记录：下一条新预约的 id 应紧接其后
    nxt = client.post("/api/v1/reservations", json=_payload(
        request_id="req-2",
        start_utc="2026-03-09T15:00:00Z",
        end_utc="2026-03-09T16:00:00Z",
    ))
    assert nxt.status_code == 201
    assert nxt.json()["reservation_id"] == first.json()["reservation_id"] + 1


def test_equivalent_timezone_offsets_are_the_same_request(client):
    first = client.post("/api/v1/reservations", json=_payload())
    assert first.status_code == 201
    # +09:00 表示的同一物理时刻，规范化后视为同一请求 → 幂等重放
    replay = client.post("/api/v1/reservations", json=_payload(
        start_utc="2026-03-09T22:00:00+09:00",
        end_utc="2026-03-09T23:00:00+09:00",
    ))
    assert replay.status_code == 200
    assert replay.json()["reservation_id"] == first.json()["reservation_id"]


def test_same_request_id_different_content_returns_409(client):
    assert client.post("/api/v1/reservations", json=_payload()).status_code == 201
    # 时间不同
    resp = client.post("/api/v1/reservations",
                       json=_payload(end_utc="2026-03-09T15:00:00Z"))
    assert resp.status_code == 409
    assert resp.json()["existing_reservation_id"] == 1
    # 参与者不同
    resp = client.post("/api/v1/reservations",
                       json=_payload(participant_ids=["alice"]))
    assert resp.status_code == 409


# ---------------------------------------------------------------------- 并发

def test_concurrent_conflicting_requests_single_winner(db_path):
    """8 个并发请求抢占同一参与者的同一时段：恰好一个成功，其余全部 409 语义。"""
    barrier = threading.Barrier(8)

    def worker(i: int):
        barrier.wait()
        return core.create_reservation(
            request_id=f"req-{i}",
            start_utc=_utc(2026, 3, 9, 13),
            end_utc=_utc(2026, 3, 9, 14),
            participant_ids=["alice"],
            db_path=db_path,
        )

    created, conflicts = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        for f in futures:
            try:
                created.append(f.result())
            except core.ReservationConflictError as exc:
                conflicts.append(exc)

    assert len(created) == 1
    assert len(conflicts) == 7
    winner = created[0][0]
    assert created[0][1] is True
    for exc in conflicts:
        # 所有冲突都指向赢家预约，且没有任何部分写入
        assert exc.conflicts == [
            {"reservation_id": winner.id, "participant_id": "alice"}
        ]


def test_concurrent_idempotent_replays_create_single_record(db_path):
    """同一 request_id + 同一内容并发重试：全部拿到同一条预约，只新建一次。"""
    barrier = threading.Barrier(8)
    kwargs = dict(
        request_id="req-same",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice", "bob"],
        db_path=db_path,
    )

    def worker():
        barrier.wait()
        return core.create_reservation(**kwargs)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: worker(), range(8)))

    ids = {r.id for r, _ in outcomes}
    assert len(ids) == 1                                    # 同一条预约
    assert sum(1 for _, created in outcomes if created) == 1  # 只新建一次
    # 数据库里确实只有这一条：下一条预约 id 紧接其后
    nxt, created = core.create_reservation(
        request_id="req-next",
        start_utc=_utc(2026, 3, 10, 13),
        end_utc=_utc(2026, 3, 10, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    assert created is True
    assert nxt.id == ids.pop() + 1


def test_concurrent_requests_via_api(db_path):
    """API 层并发：同一参与者同一时段，恰好一个 201，其余 409。"""

    def worker(i: int):
        client = TestClient(app)
        return client.post("/api/v1/reservations", json=_payload(
            request_id=f"req-{i}",
            participant_ids=["alice"],
        ))

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(worker, range(6)))

    assert sorted(r.status_code for r in responses) == [201, 409, 409, 409, 409, 409]


# ---------------------------------------------------------------------- 改期

def _put(client, reservation_id, **overrides):
    payload = {
        "expected_version": 1,
        "start_utc": "2026-03-09T15:00:00Z",
        "end_utc": "2026-03-09T16:00:00Z",
        "participant_ids": ["alice", "bob"],
    }
    payload.update(overrides)
    return client.put(f"/api/v1/reservations/{reservation_id}", json=payload)


def test_create_response_includes_version_1(client):
    resp = client.post("/api/v1/reservations", json=_payload())
    assert resp.status_code == 201
    assert resp.json()["version"] == 1


def test_reschedule_success_replaces_window_and_bumps_version(client):
    created = client.post("/api/v1/reservations", json=_payload()).json()
    rid = created["reservation_id"]
    assert created["version"] == 1

    resp = _put(client, rid, start_utc="2026-03-09T16:00:00Z",
                end_utc="2026-03-09T16:30:00Z")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reservation_id"] == rid
    assert data["start_utc"] == "2026-03-09T16:00:00Z"
    assert data["end_utc"] == "2026-03-09T16:30:00Z"
    assert data["participants"] == ["alice", "bob"]
    assert data["version"] == 2
    # request_id / created_at 保持不变
    assert data["request_id"] == created["request_id"]
    assert data["created_at"] == created["created_at"]

    # 第二次改期须基于新版本
    resp = _put(client, rid, expected_version=2,
                start_utc="2026-03-09T17:00:00Z",
                end_utc="2026-03-09T17:30:00Z")
    assert resp.status_code == 200, resp.text
    assert resp.json()["version"] == 3


def test_reschedule_normalizes_participants(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    resp = _put(client, rid, participant_ids=["carol", "alice", "carol"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["participants"] == ["alice", "carol"]


def test_reschedule_unknown_id_returns_404(client):
    resp = _put(client, 999)
    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


def test_reschedule_stale_version_returns_409_with_current_version(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    # 第一次改期成功，版本推进到 2
    assert _put(client, rid).status_code == 200
    # 仍用 expected_version=1 → 409 并告知当前版本
    resp = _put(client, rid, expected_version=1,
                start_utc="2026-03-09T18:00:00Z",
                end_utc="2026-03-09T18:30:00Z")
    assert resp.status_code == 409
    assert resp.json()["current_version"] == 2

    # 带上正确版本后成功
    resp = _put(client, rid, expected_version=2,
                start_utc="2026-03-09T18:00:00Z",
                end_utc="2026-03-09T18:30:00Z")
    assert resp.status_code == 200, resp.text
    assert resp.json()["version"] == 3


def test_reschedule_conflict_excludes_self(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    # 与自身旧窗口相同：排除自身后无冲突
    resp = _put(client, rid,
                start_utc="2026-03-09T13:00:00Z",
                end_utc="2026-03-09T14:00:00Z")
    assert resp.status_code == 200, resp.text
    assert resp.json()["version"] == 2


def test_reschedule_adjacent_to_self_is_fine(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    # 紧邻自身旧窗口（半开区间相邻不冲突）
    resp = _put(client, rid,
                start_utc="2026-03-09T14:00:00Z",
                end_utc="2026-03-09T15:00:00Z")
    assert resp.status_code == 200, resp.text
    assert resp.json()["version"] == 2


def test_reschedule_conflict_with_other_reservation_returns_409(client):
    first = client.post("/api/v1/reservations", json=_payload(
        request_id="req-A",
        start_utc="2026-03-09T13:00:00Z",
        end_utc="2026-03-09T14:00:00Z",
        participant_ids=["alice"],
    )).json()
    second = client.post("/api/v1/reservations", json=_payload(
        request_id="req-B",
        start_utc="2026-03-09T16:00:00Z",
        end_utc="2026-03-09T17:00:00Z",
        participant_ids=["bob"],
    )).json()
    # 把第二场改到 alice 另一场的时段（新参与者集合包含 alice）→ 冲突明细指向 first
    resp = _put(client, second["reservation_id"],
                start_utc="2026-03-09T13:30:00Z",
                end_utc="2026-03-09T13:45:00Z",
                participant_ids=["alice", "carol"])
    assert resp.status_code == 409
    assert resp.json()["conflicts"] == [
        {"reservation_id": first["reservation_id"], "participant_id": "alice"}
    ]


def test_reschedule_conflict_does_not_mutate_reservation_or_occupancy(client, db_path):
    created = client.post("/api/v1/reservations", json=_payload(
        participant_ids=["alice", "bob"])).json()
    rid = created["reservation_id"]
    # 占用 15:00–16:00 的另一场会
    assert client.post("/api/v1/reservations", json=_payload(
        request_id="req-other",
        start_utc="2026-03-09T15:00:00Z",
        end_utc="2026-03-09T16:00:00Z",
        participant_ids=["carol"],
    )).status_code == 201

    # 新集合包含 carol → 与 carol 自己 15:00–16:00 的另一场冲突；dave 是顺带参与者
    resp = _put(client, rid, participant_ids=["carol", "dave"])
    assert resp.status_code == 409

    # 预约本身未变：时间、参与者、版本
    got = client.post("/api/v1/reservations", json=_payload()).json()  # 幂等重放
    assert got["reservation_id"] == rid
    assert got["start_utc"] == "2026-03-09T13:00:00Z"
    assert got["participants"] == ["alice", "bob"]
    assert got["version"] == 1

    # 旧参与者占用仍在；dave 未留下残留占用
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-dave",
        start_utc="2026-03-09T15:30:00Z",
        end_utc="2026-03-09T15:45:00Z",
        participant_ids=["dave"],
    ))
    assert resp.status_code == 201, resp.text
    # 而 alice 在旧窗口仍被占用
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-alice-check",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    ))
    assert resp.status_code == 409


def test_reschedule_frees_old_window_and_occupies_new(client):
    rid = client.post("/api/v1/reservations", json=_payload(
        participant_ids=["alice"])).json()["reservation_id"]
    assert _put(client, rid,
                start_utc="2026-03-09T15:00:00Z",
                end_utc="2026-03-09T16:00:00Z").status_code == 200
    # 旧窗口 13:00–14:00 已释放：他人可预约
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-moved-in",
        start_utc="2026-03-09T13:00:00Z",
        end_utc="2026-03-09T14:00:00Z",
        participant_ids=["alice"],
    ))
    assert resp.status_code == 201, resp.text
    # 新窗口 15:00–16:00 已占用
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-blocked",
        start_utc="2026-03-09T15:30:00Z",
        end_utc="2026-03-09T15:45:00Z",
        participant_ids=["alice"],
    ))
    assert resp.status_code == 409


def test_reschedule_replaces_participant_set(client):
    rid = client.post("/api/v1/reservations", json=_payload(
        participant_ids=["alice", "bob"])).json()["reservation_id"]
    resp = _put(client, rid, participant_ids=["carol"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["participants"] == ["carol"]
    # alice/bob 旧占用被释放
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-alice-free",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    ))
    assert resp.status_code == 201, resp.text
    # carol 在新窗口被占用
    resp = client.post("/api/v1/reservations", json=_payload(
        request_id="req-carol-busy",
        start_utc="2026-03-09T15:30:00Z",
        end_utc="2026-03-09T15:45:00Z",
        participant_ids=["carol"],
    ))
    assert resp.status_code == 409


def test_reschedule_validation_errors_return_422(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    # end <= start
    assert _put(client, rid,
                start_utc="2026-03-09T15:00:00Z",
                end_utc="2026-03-09T15:00:00Z").status_code == 422
    # 无时区偏移
    assert _put(client, rid,
                start_utc="2026-03-09T15:00:00").status_code == 422
    # 空参与者
    assert _put(client, rid, participant_ids=[]).status_code == 422
    # expected_version < 1
    assert _put(client, rid, expected_version=0).status_code == 422


def test_concurrent_reschedules_single_winner(db_path):
    """8 个并发改期请求（同一 expected_version）：恰一个成功，其余 409 版本冲突。

    各请求目标窗口互不重叠，因此唯一的失败原因是版本过期，
    证明“版本校验 + 替换”在 BEGIN IMMEDIATE 下被串行化。
    """
    reservation, _ = core.create_reservation(
        request_id="req-base",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )

    def worker(i: int):
        return core.reschedule_reservation(
            reservation_id=reservation.id,
            expected_version=1,
            start_utc=_utc(2026, 3, 10 + i, 13),
            end_utc=_utc(2026, 3, 10 + i, 14),
            participant_ids=["alice"],
            db_path=db_path,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        results = []
        for f in futures:
            try:
                results.append(("ok", f.result()))
            except core.VersionConflictError as exc:
                results.append(("version", exc))

    oks = [r for kind, r in results if kind == "ok"]
    stale = [r for kind, r in results if kind == "version"]
    assert len(oks) == 1
    assert len(stale) == 7
    assert oks[0].version == 2
    assert {e.current_version for e in stale} == {2}


def test_concurrent_reschedules_via_api(db_path):
    """API 层并发改期：同一 expected_version 下恰好一个 200，其余 409。"""
    client = TestClient(app)
    rid = client.post("/api/v1/reservations", json=_payload(
        participant_ids=["alice"])).json()["reservation_id"]

    def worker(i: int):
        c = TestClient(app)
        return c.put(f"/api/v1/reservations/{rid}", json={
            "expected_version": 1,
            "start_utc": f"2026-03-{20 + i:02d}T13:00:00Z",
            "end_utc": f"2026-03-{20 + i:02d}T14:00:00Z",
            "participant_ids": ["alice"],
        })

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(worker, range(6)))

    assert sorted(r.status_code for r in responses) == [200, 409, 409, 409, 409, 409]
    body = client.put(f"/api/v1/reservations/{rid}", json={
        "expected_version": 2,
        "start_utc": "2026-04-01T13:00:00Z",
        "end_utc": "2026-04-01T14:00:00Z",
        "participant_ids": ["alice"],
    })
    assert body.status_code == 200, body.text
    assert body.json()["version"] == 3


# ---------------------------------------------------------------------- 取消

def _delete(client, reservation_id, expected_version):
    return client.delete(
        f"/api/v1/reservations/{reservation_id}"
        f"?expected_version={expected_version}"
    )


def test_cancel_success_soft_deletes_bumps_version_and_frees_slot(client):
    created = client.post("/api/v1/reservations", json=_payload()).json()
    rid = created["reservation_id"]

    resp = _delete(client, rid, 1)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reservation_id"] == rid
    assert data["request_id"] == created["request_id"]
    assert data["status"] == "canceled"
    assert data["version"] == 2
    assert data["canceled_at"]                      # 记录了取消时刻
    # 原时间/参与者快照仍在响应中
    assert data["start_utc"] == "2026-03-09T13:00:00Z"
    assert data["participants"] == ["alice", "bob"]
    assert data["created_at"] == created["created_at"]

    # 占用已删除：同一参与者的同一时段立即可再次预约
    again = client.post("/api/v1/reservations", json=_payload(
        request_id="req-after-cancel"))
    assert again.status_code == 201, again.text
    assert again.json()["reservation_id"] != rid
    assert again.json()["status"] == "active"
    assert again.json()["version"] == 1


def test_cancel_removes_all_participant_occupancy(client):
    rid = client.post("/api/v1/reservations", json=_payload(
        participant_ids=["alice", "bob", "carol"])).json()["reservation_id"]
    assert _delete(client, rid, 1).status_code == 200

    # 三位参与者在原时段都应立即可约
    for pid in ("alice", "bob", "carol"):
        resp = client.post("/api/v1/reservations", json=_payload(
            request_id=f"req-{pid}",
            participant_ids=[pid]))
        assert resp.status_code == 201, (pid, resp.text)


def test_cancel_unknown_id_returns_404(client):
    resp = _delete(client, 999, 1)
    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


def test_cancel_stale_version_returns_409_with_current_version(client):
    rid = client.post("/api/v1/reservations", json=_payload(
        participant_ids=["alice"])).json()["reservation_id"]
    # 先改期到不冲突的窗口，把版本推进到 2（旧窗口随之释放）
    assert _put(client, rid,
                start_utc="2026-03-10T13:00:00Z",
                end_utc="2026-03-10T14:00:00Z").status_code == 200
    # 仍用旧版本取消 → 409 + current_version
    resp = _delete(client, rid, 1)
    assert resp.status_code == 409
    assert resp.json()["current_version"] == 2

    # 预约仍然有效：新窗口仍被 alice 占用
    blocked = client.post("/api/v1/reservations", json=_payload(
        request_id="req-blocked",
        start_utc="2026-03-10T13:30:00Z",
        end_utc="2026-03-10T13:45:00Z",
        participant_ids=["alice"]))
    assert blocked.status_code == 409

    # 带上正确版本可以取消成功
    ok = _delete(client, rid, 2)
    assert ok.status_code == 200, ok.text
    assert ok.json()["version"] == 3
    assert ok.json()["status"] == "canceled"


def test_cancel_idempotent_retry_returns_same_result(client):
    created = client.post("/api/v1/reservations", json=_payload()).json()
    rid = created["reservation_id"]

    first = _delete(client, rid, 1)
    assert first.status_code == 200
    # 相同 reservation_id + 相同原 expected_version 的重试 → 幂等返回同一结果
    retry = _delete(client, rid, 1)
    assert retry.status_code == 200, retry.text
    assert retry.json() == first.json()
    # canceled_at / version 不随重试变化（没有重复写入）
    assert retry.json()["version"] == 2
    assert retry.json()["canceled_at"] == first.json()["canceled_at"]

    # 连续多次重试结果一致
    assert _delete(client, rid, 1).json() == first.json()


def test_cancel_after_cancel_other_old_version_returns_409(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    assert _delete(client, rid, 1).status_code == 200
    # 取消后当前版本为 2：用任何非“取消时版本”（此处只能是 2）重试 → 409
    resp = _delete(client, rid, 2)
    assert resp.status_code == 409
    assert resp.json()["current_version"] == 2


def test_cancel_after_reschedule_uses_rescheduled_version(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    assert _put(client, rid).status_code == 200            # version -> 2
    # 用 v=1 取消既不是有效版本也不是“取消时版本” → 409
    assert _delete(client, rid, 1).status_code == 409
    # v=2 取消成功 → version 3；之后 v=2 的重试幂等
    assert _delete(client, rid, 2).json()["version"] == 3
    retry = _delete(client, rid, 2)
    assert retry.status_code == 200
    assert retry.json()["version"] == 3


def test_put_after_cancel_returns_409_and_does_not_resurrect(client):
    created = client.post("/api/v1/reservations", json=_payload()).json()
    rid = created["reservation_id"]
    assert _delete(client, rid, 1).status_code == 200

    # 即使携带“当前版本 2”，改期也必须被拒绝
    resp = _put(client, rid, expected_version=2)
    assert resp.status_code == 409
    assert resp.json()["current_version"] == 2
    # 用取消前的旧版本同样拒绝
    assert _put(client, rid, expected_version=1).status_code == 409

    # 再次取消仍是幂等的取消结果，预约未被改期复活
    again = _delete(client, rid, 1)
    assert again.status_code == 200
    assert again.json()["status"] == "canceled"
    assert again.json()["version"] == 2


def test_cancel_does_not_reopen_occupancy_via_reschedule(client):
    # 取消后尝试改期到一个与其他有效预约冲突的窗口，也必须 409 而不是先复活再冲突
    client.post("/api/v1/reservations", json=_payload(
        request_id="req-A", participant_ids=["alice"]))
    rid = client.post("/api/v1/reservations", json=_payload(
        request_id="req-B", start_utc="2026-03-09T16:00:00Z",
        end_utc="2026-03-09T17:00:00Z",
        participant_ids=["bob"])).json()["reservation_id"]
    assert _delete(client, rid, 1).status_code == 200
    resp = _put(client, rid, expected_version=2,
                start_utc="2026-03-09T13:30:00Z",
                end_utc="2026-03-09T13:45:00Z",
                participant_ids=["alice"])
    assert resp.status_code == 409
    assert "canceled" in resp.json()["detail"]


def test_cancel_then_replay_same_request_id_returns_409(client):
    created = client.post("/api/v1/reservations", json=_payload()).json()
    rid = created["reservation_id"]
    assert _delete(client, rid, 1).status_code == 200
    # 取消后用原 request_id 重放（相同内容）不得复活已取消预约
    replay = client.post("/api/v1/reservations", json=_payload())
    assert replay.status_code == 409
    assert replay.json()["existing_reservation_id"] == rid


def test_cancel_requires_expected_version_query_param(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    resp = client.delete(f"/api/v1/reservations/{rid}")
    assert resp.status_code == 422  # 缺少必填查询参数


def test_cancel_non_positive_expected_version_returns_422(client):
    rid = client.post("/api/v1/reservations", json=_payload()).json()["reservation_id"]
    resp = client.delete(f"/api/v1/reservations/{rid}?expected_version=0")
    assert resp.status_code == 422


def test_cancel_core_success_and_idempotency(db_path):
    created, _ = core.create_reservation(
        request_id="req-1",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice", "bob"],
        db_path=db_path,
    )
    canceled = core.cancel_reservation(created.id, 1, db_path=db_path)
    assert canceled.status == "canceled"
    assert canceled.version == 2
    assert canceled.canceled_at is not None

    # 幂等：同版本再次取消返回同一结果
    replay = core.cancel_reservation(created.id, 1, db_path=db_path)
    assert replay == canceled

    # 其他旧版本 → VersionConflictError 携带当前版本
    with pytest.raises(core.VersionConflictError) as exc:
        core.cancel_reservation(created.id, 2, db_path=db_path)
    assert exc.value.current_version == 2

    # 不存在的预约 → NotFound
    with pytest.raises(core.ReservationNotFoundError):
        core.cancel_reservation(999, 1, db_path=db_path)


def test_cancel_core_frees_slot_and_put_rejected(db_path):
    created, _ = core.create_reservation(
        request_id="req-1",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    core.cancel_reservation(created.id, 1, db_path=db_path)

    # 时段立即可再次预约（新 request_id）
    other, created_new = core.create_reservation(
        request_id="req-2",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    assert created_new is True
    assert other.id != created.id

    # 改期已取消预约 → ReservationCanceledError（携带当前版本）
    with pytest.raises(core.ReservationCanceledError) as exc:
        core.reschedule_reservation(
            reservation_id=created.id,
            expected_version=2,
            start_utc=_utc(2026, 3, 10, 13),
            end_utc=_utc(2026, 3, 10, 14),
            participant_ids=["alice"],
            db_path=db_path,
        )
    assert exc.value.current_version == 2


def test_cancel_core_failure_rolls_back(db_path):
    # 版本不匹配时取消不得改动预约状态/版本，占用保持
    created, _ = core.create_reservation(
        request_id="req-1",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    with pytest.raises(core.VersionConflictError):
        core.cancel_reservation(created.id, 7, db_path=db_path)

    # 占用仍在：同参与者重叠时段无法再约 → 冲突（证明未释放、状态未变）
    with pytest.raises(core.ReservationConflictError):
        core.create_reservation(
            request_id="req-2",
            start_utc=_utc(2026, 3, 9, 13, 30),
            end_utc=_utc(2026, 3, 9, 13, 45),
            participant_ids=["alice"],
            db_path=db_path,
        )


def test_concurrent_cancels_single_write_all_idempotent(db_path):
    """8 个并发取消（同一 expected_version）：恰一个执行软删除，全部拿到同一结果。

    取消按“相同 reservation_id + 相同 expected_version”幂等：
    落败者在写锁内看到已取消状态且版本正好为 current-1，转走幂等返回。
    """
    import sqlite3

    created, _ = core.create_reservation(
        request_id="req-base",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return core.cancel_reservation(created.id, 1, db_path=db_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: worker(), range(8)))

    # 全部返回同一条已取消预约（幂等），状态/版本/canceled_at 完全一致
    assert len({(r.id, r.version, r.status, r.canceled_at) for r in results}) == 1
    assert results[0].status == "canceled"
    assert results[0].version == 2

    # 占用恰好被清空（只删除一次也为空），取消行只有一条
    conn = sqlite3.connect(db_path)
    try:
        status = conn.execute(
            "SELECT status, version FROM reservations WHERE id = ?",
            (created.id,)).fetchone()
        n_occ = conn.execute(
            "SELECT COUNT(*) FROM reservation_participants"
            " WHERE reservation_id = ?", (created.id,)).fetchone()[0]
    finally:
        conn.close()
    assert status == ("canceled", 2)
    assert n_occ == 0


def test_concurrent_cancel_and_reschedule_single_winner(db_path):
    """同版本的并发取消与改期：写锁串行化，恰一个生效；落败者得到 409 语义。"""
    created, _ = core.create_reservation(
        request_id="req-base",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    barrier = threading.Barrier(2)

    def do_cancel():
        barrier.wait()
        return core.cancel_reservation(created.id, 1, db_path=db_path)

    def do_reschedule():
        barrier.wait()
        return core.reschedule_reservation(
            reservation_id=created.id,
            expected_version=1,
            start_utc=_utc(2026, 3, 11, 13),
            end_utc=_utc(2026, 3, 11, 14),
            participant_ids=["alice"],
            db_path=db_path,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_cancel = pool.submit(do_cancel)
        f_move = pool.submit(do_reschedule)
        outcomes = []
        for f in (f_cancel, f_move):
            try:
                outcomes.append(("ok", f.result()))
            except (core.VersionConflictError,
                    core.ReservationCanceledError) as exc:
                outcomes.append(("conflict", exc))

    kinds = [k for k, _ in outcomes]
    assert kinds.count("ok") == 1
    assert kinds.count("conflict") == 1
    # 最终状态自洽：若取消赢则预约 canceled；若改期赢则预约 active@v2
    winner = [r for k, r in outcomes if k == "ok"][0]
    if winner.status == "canceled":
        # 改期落败：取消后再改期仍被拒
        with pytest.raises(core.ReservationCanceledError):
            core.reschedule_reservation(
                reservation_id=created.id, expected_version=2,
                start_utc=_utc(2026, 3, 12, 13),
                end_utc=_utc(2026, 3, 12, 14),
                participant_ids=["alice"], db_path=db_path)
    else:
        assert winner.version == 2
        # 改期赢后，旧版本取消是版本冲突（不是幂等取消）
        with pytest.raises(core.VersionConflictError):
            core.cancel_reservation(created.id, 1, db_path=db_path)


# -------------------------------------------------- 既有数据库无损迁移 version

_LEGACY_SCHEMA = """\
CREATE TABLE reservations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id       TEXT NOT NULL UNIQUE,
    start_utc        TEXT NOT NULL,
    end_utc          TEXT NOT NULL,
    participants_key TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE reservation_participants (
    reservation_id INTEGER NOT NULL REFERENCES reservations(id) ON DELETE CASCADE,
    participant_id TEXT NOT NULL,
    start_utc      TEXT NOT NULL,
    end_utc        TEXT NOT NULL,
    PRIMARY KEY (reservation_id, participant_id)
);
CREATE INDEX idx_rp_participant_window
    ON reservation_participants (participant_id, start_utc, end_utc);
"""


def test_startup_backfills_version_1_for_legacy_data(tmp_path, monkeypatch):
    import sqlite3

    path = str(tmp_path / "legacy.db")
    # 模拟旧版本应用写入的既有数据（无 version 列）
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_LEGACY_SCHEMA)
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, created_at)"
            " VALUES ('old-req', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z', '[\"alice\", \"bob\"]',"
            " '2026-03-01T00:00:00.000000Z')")
        conn.execute(
            "INSERT INTO reservation_participants"
            " (reservation_id, participant_id, start_utc, end_utc)"
            " VALUES (1, 'alice', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z')")
        conn.execute(
            "INSERT INTO reservation_participants"
            " (reservation_id, participant_id, start_utc, end_utc)"
            " VALUES (1, 'bob', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z')")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    # 启动时迁移
    db_module.init_db(path)

    # 迁移幂等：再跑一次也不报错
    db_module.init_db(path)

    # 核心层读到既有预约，version 补齐为 1，数据无损
    got = core.create_reservation(
        request_id="old-req",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["bob", "alice"],
        db_path=path,
    )[0]
    assert got.id == 1
    assert got.version == 1
    assert got.participants == ("alice", "bob")

    # 基于补齐的 version=1 可以改期成功
    moved = core.reschedule_reservation(
        reservation_id=1,
        expected_version=1,
        start_utc=_utc(2026, 3, 9, 16),
        end_utc=_utc(2026, 3, 9, 17),
        participant_ids=["alice", "bob"],
        db_path=path,
    )
    assert moved.version == 2

    # 旧占用已随改期迁移到新窗口
    other, created = core.create_reservation(
        request_id="new-req",
        start_utc=_utc(2026, 3, 9, 13, 30),
        end_utc=_utc(2026, 3, 9, 13, 45),
        participant_ids=["alice"],
        db_path=path,
    )
    assert created is True


_V1_SCHEMA = """\
CREATE TABLE reservations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id       TEXT NOT NULL UNIQUE,
    start_utc        TEXT NOT NULL,
    end_utc          TEXT NOT NULL,
    participants_key TEXT NOT NULL,
    version          INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL
);
CREATE TABLE reservation_participants (
    reservation_id INTEGER NOT NULL REFERENCES reservations(id) ON DELETE CASCADE,
    participant_id TEXT NOT NULL,
    start_utc      TEXT NOT NULL,
    end_utc        TEXT NOT NULL,
    PRIMARY KEY (reservation_id, participant_id)
);
CREATE INDEX idx_rp_participant_window
    ON reservation_participants (participant_id, start_utc, end_utc);
"""


def test_startup_backfills_status_columns_losslessly(tmp_path, monkeypatch):
    """已有 version 列但缺少 status/canceled_at 的库：无损补列后既有预约仍可用。"""
    import sqlite3

    path = str(tmp_path / "v1.db")
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_V1_SCHEMA)
        # 一条从未改期（version=1）与一条已改期（version=3）的既有预约
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, version, created_at)"
            " VALUES ('r1', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z', '[\"alice\"]', 1,"
            " '2026-03-01T00:00:00.000000Z')")
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, version, created_at)"
            " VALUES ('r2', '2026-03-09T16:00:00.000000Z',"
            " '2026-03-09T17:00:00.000000Z', '[\"bob\"]', 3,"
            " '2026-03-02T00:00:00.000000Z')")
        conn.execute(
            "INSERT INTO reservation_participants"
            " (reservation_id, participant_id, start_utc, end_utc)"
            " VALUES (1, 'alice', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z')")
        conn.execute(
            "INSERT INTO reservation_participants"
            " (reservation_id, participant_id, start_utc, end_utc)"
            " VALUES (2, 'bob', '2026-03-09T16:00:00.000000Z',"
            " '2026-03-09T17:00:00.000000Z')")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    db_module.init_db(path)
    db_module.init_db(path)  # 迁移可重入

    # 既有数据无损：version 保留、状态补为 active、canceled_at 为 NULL
    r1 = core.cancel_reservation(1, 1, db_path=path)
    assert r1.status == "canceled" and r1.version == 2
    assert r1.canceled_at is not None
    # r2 已改过两次期，须用其既有版本 3 才能取消
    with pytest.raises(core.VersionConflictError) as exc:
        core.cancel_reservation(2, 1, db_path=path)
    assert exc.value.current_version == 3
    r2 = core.cancel_reservation(2, 3, db_path=path)
    assert r2.status == "canceled" and r2.version == 4

    # 既有占用随取消全部释放
    conn = sqlite3.connect(path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM reservation_participants").fetchone()[0]
    finally:
        conn.close()
    assert n == 0


# ---------------------------------------------------------- 聚合接口回归

def test_aggregate_endpoint_still_works(client):
    resp = client.post("/api/v1/availability/aggregate", json={
        "start_date": "2026-03-09",
        "end_date": "2026-03-09",
        "meeting_minutes": 60,
        "min_attendees": 1,
        "participants": [
            {"id": "a", "timezone": "UTC",
             "availability": [{"weekday": 1, "start_local": "09:00", "end_local": "17:00"}]},
        ],
    })
    assert resp.status_code == 200
    assert len(resp.json()["segments"]) == 1
