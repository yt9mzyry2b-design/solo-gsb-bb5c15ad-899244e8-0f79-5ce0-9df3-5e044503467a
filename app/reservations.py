"""会议预约核心逻辑：幂等创建、改期、取消与单事务冲突检测。

- 参与者 ID 去重并按字典序排序后，作为规范化请求内容参与幂等比较；
- 区间一律按半开 ``[start, end)`` 处理：首尾相接（相邻）不算冲突；
- 冲突检查与写入在同一个 ``BEGIN IMMEDIATE`` 事务内完成：
  SQLite 在事务开始时即取写锁，并发请求被串行化；
  任一参与者与已有预约重叠都会整体回滚，不会留下部分写入。
- 每条预约带单调递增的 ``version``（创建为 1）；改期/取消采用乐观并发控制：
  请求须携带 ``expected_version``，匹配才执行，成功后 ``version`` 加 1。
- 取消为**软删除**：预约行 ``status`` 置为 ``canceled`` 并记录 ``canceled_at``、
  ``version`` 加 1，同时删除全部参与者占用（时段立即可再次预约）；
  已取消的预约不能改期（不会复活），取消请求本身按
  “相同 reservation_id + 相同 expected_version”幂等。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import db

# 定宽 UTC 文本格式：字典序即时间序，可直接用于 SQL 范围比较
_CANON_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# 预约状态：active=有效；canceled=已取消（软删除，参与者占用已释放）
STATUS_ACTIVE = "active"
STATUS_CANCELED = "canceled"


class ReservationConflictError(Exception):
    """至少一位参与者与已有预约存在半开区间重叠。"""

    def __init__(self, conflicts: list[dict[str, object]]) -> None:
        self.conflicts = conflicts  # [{"reservation_id": ..., "participant_id": ...}]
        pretty = ", ".join(
            f"participant {c['participant_id']!r} overlaps reservation #{c['reservation_id']}"
            for c in conflicts
        )
        super().__init__(f"reservation conflict: {pretty}")


class RequestIdConflictError(Exception):
    """request_id 已被占用（内容不同，或预约已取消而无法幂等重放）。"""

    def __init__(self, existing_reservation_id: int,
                 canceled: bool = False) -> None:
        self.existing_reservation_id = existing_reservation_id
        self.canceled = canceled
        if canceled:
            message = (
                f"request_id belongs to canceled reservation "
                f"#{existing_reservation_id}; a canceled reservation is not "
                "reopened by replay"
            )
        else:
            message = (
                f"request_id is already used by reservation "
                f"#{existing_reservation_id} with different content"
            )
        super().__init__(message)


class ReservationNotFoundError(Exception):
    """指定 ID 的预约不存在。"""

    def __init__(self, reservation_id: int) -> None:
        self.reservation_id = reservation_id
        super().__init__(f"reservation #{reservation_id} not found")


class ReservationCanceledError(Exception):
    """预约已取消（软删除）：不能改期，取消幂等重放由取消接口自行处理。"""

    def __init__(self, reservation_id: int, current_version: int) -> None:
        self.reservation_id = reservation_id
        self.current_version = current_version
        super().__init__(
            f"reservation #{reservation_id} has been canceled and can no "
            "longer be rescheduled"
        )


class VersionConflictError(Exception):
    """expected_version 与预约当前版本不一致（乐观锁）。"""

    def __init__(self, reservation_id: int, current_version: int) -> None:
        self.reservation_id = reservation_id
        self.current_version = current_version
        super().__init__(
            f"expected_version does not match current version of reservation "
            f"#{reservation_id}"
        )


@dataclass(frozen=True)
class Reservation:
    id: int
    request_id: str
    start_utc: datetime
    end_utc: datetime
    participants: tuple[str, ...]
    created_at: datetime
    version: int = 1
    status: str = STATUS_ACTIVE
    canceled_at: datetime | None = None


_RESERVATION_COLUMNS = (
    "id, request_id, start_utc, end_utc, participants_key, version, status,"
    " canceled_at, created_at"
)


def _canon(dt: datetime) -> str:
    """统一成定宽 UTC 字符串，保证字典序与时间序一致。"""
    return dt.astimezone(timezone.utc).strftime(_CANON_FMT)


def _parse(text: str) -> datetime:
    return datetime.strptime(text, _CANON_FMT).replace(tzinfo=timezone.utc)


def normalize_participants(participant_ids: Iterable[str]) -> tuple[str, ...]:
    """去重 + 字典序排序，得到规范化参与者列表。"""
    return tuple(sorted(set(participant_ids)))


def _row_to_reservation(row: sqlite3.Row) -> Reservation:
    return Reservation(
        id=row["id"],
        request_id=row["request_id"],
        start_utc=_parse(row["start_utc"]),
        end_utc=_parse(row["end_utc"]),
        participants=tuple(json.loads(row["participants_key"])),
        created_at=_parse(row["created_at"]),
        version=row["version"],
        status=row["status"],
        canceled_at=_parse(row["canceled_at"]) if row["canceled_at"] else None,
    )


def create_reservation(
    request_id: str,
    start_utc: datetime,
    end_utc: datetime,
    participant_ids: Iterable[str],
    db_path: str | None = None,
) -> tuple[Reservation, bool]:
    """创建预约，返回 ``(预约, 是否新建)``。

    - ``request_id`` 已存在且规范化内容（起止时刻 + 去重排序后的参与者）相同
      → 返回原预约，``created=False``，不新增记录；
    - ``request_id`` 已存在但内容不同 → :class:`RequestIdConflictError`；
    - 任一参与者与已有预约半开区间重叠 → :class:`ReservationConflictError`，
      整体回滚，不留下任何部分写入。
    """
    participants = normalize_participants(participant_ids)
    start_s = _canon(start_utc)
    end_s = _canon(end_utc)
    participants_key = json.dumps(list(participants), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：事务开始即持有数据库写锁，
        # 使“检查 + 写入”对并发请求原子（并发写者被串行化）。
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                f"SELECT {_RESERVATION_COLUMNS}"
                " FROM reservations WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is not None:
                if row["status"] == STATUS_CANCELED:
                    # 取消为终态：重放不得“复活”已取消预约（其占用已释放），
                    # 原 request_id 也不再接受同内容重放，避免 500（UNIQUE）歧义。
                    raise RequestIdConflictError(row["id"], canceled=True)
                if (row["start_utc"] == start_s
                        and row["end_utc"] == end_s
                        and row["participants_key"] == participants_key):
                    conn.execute("COMMIT")
                    return _row_to_reservation(row), False
                raise RequestIdConflictError(row["id"])

            conflicts: list[dict[str, object]] = []
            for pid in participants:
                # 半开区间重叠：existing.start < new.end 且 new.start < existing.end
                # （相邻区间因等号不成立而不会命中）
                for hit in conn.execute(
                    "SELECT reservation_id, participant_id"
                    " FROM reservation_participants"
                    " WHERE participant_id = ? AND start_utc < ? AND ? < end_utc"
                    " ORDER BY reservation_id",
                    (pid, end_s, start_s),
                ):
                    conflicts.append({
                        "reservation_id": hit["reservation_id"],
                        "participant_id": hit["participant_id"],
                    })
            if conflicts:
                raise ReservationConflictError(conflicts)

            now_s = _canon(datetime.now(timezone.utc))
            cursor = conn.execute(
                "INSERT INTO reservations"
                " (request_id, start_utc, end_utc, participants_key, version, created_at)"
                " VALUES (?, ?, ?, ?, 1, ?)",
                (request_id, start_s, end_s, participants_key, now_s),
            )
            reservation_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO reservation_participants"
                " (reservation_id, participant_id, start_utc, end_utc)"
                " VALUES (?, ?, ?, ?)",
                [(reservation_id, pid, start_s, end_s) for pid in participants],
            )
            conn.execute("COMMIT")
            return Reservation(
                id=reservation_id,
                request_id=request_id,
                start_utc=_parse(start_s),
                end_utc=_parse(end_s),
                participants=participants,
                created_at=_parse(now_s),
                version=1,
            ), True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def reschedule_reservation(
    reservation_id: int,
    expected_version: int,
    start_utc: datetime,
    end_utc: datetime,
    participant_ids: Iterable[str],
    db_path: str | None = None,
) -> Reservation:
    """改期：原子替换预约时间与参与者占用，成功后 ``version`` 加 1。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚，
    预约行与参与者占用均保持原状：

    1. 预约不存在 → :class:`ReservationNotFoundError`（API 层映射为 404）；
    2. 预约已取消（软删除终态）→ :class:`ReservationCanceledError`（409，
       携带当前版本；改期不得复活已取消预约）；
    3. ``expected_version`` 与当前版本不一致 → :class:`VersionConflictError`
       （409，携带 ``current_version``）；
    4. 半开区间冲突检查：沿用创建时的重叠判定，但通过
       ``reservation_id != ?`` **排除自身**（自身旧占用不算冲突）；
       有冲突 → :class:`ReservationConflictError`（409 + 冲突明细）。

    成功时更新 ``reservations`` 行（时间、规范化参与者、version+1）并整体
    替换 ``reservation_participants`` 占用；``request_id`` 与 ``created_at``
    保持不变。
    """
    participants = normalize_participants(participant_ids)
    start_s = _canon(start_utc)
    end_s = _canon(end_utc)
    participants_key = json.dumps(list(participants), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：与创建路径一致，事务开始即持写锁，
        # “存在性 + 版本 + 冲突检查 + 替换” 对并发请求原子。
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                f"SELECT {_RESERVATION_COLUMNS} FROM reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise ReservationNotFoundError(reservation_id)

            if row["status"] == STATUS_CANCELED:
                # 取消是终态：任何改期（即使版本恰好匹配）一律拒绝，不复活预约
                raise ReservationCanceledError(reservation_id, row["version"])

            current_version = row["version"]
            if expected_version != current_version:
                raise VersionConflictError(reservation_id, current_version)

            conflicts: list[dict[str, object]] = []
            for pid in participants:
                # 半开区间重叠（同创建路径），reservation_id != ? 排除自身：
                # 改期前自己在该参与者上的旧占用不应被视为冲突。
                for hit in conn.execute(
                    "SELECT reservation_id, participant_id"
                    " FROM reservation_participants"
                    " WHERE participant_id = ? AND start_utc < ? AND ? < end_utc"
                    " AND reservation_id != ?"
                    " ORDER BY reservation_id",
                    (pid, end_s, start_s, reservation_id),
                ):
                    conflicts.append({
                        "reservation_id": hit["reservation_id"],
                        "participant_id": hit["participant_id"],
                    })
            if conflicts:
                raise ReservationConflictError(conflicts)

            conn.execute(
                "UPDATE reservations"
                " SET start_utc = ?, end_utc = ?, participants_key = ?,"
                "     version = version + 1"
                " WHERE id = ?",
                (start_s, end_s, participants_key, reservation_id),
            )
            # 参与者集合可能变化：先删除自身全部旧占用（子表行），再插入新占用。
            # 子表删除不会反向影响 reservations 父表行；
            # 两步与 UPDATE 同事务提交，失败即整体回滚。
            conn.execute(
                "DELETE FROM reservation_participants WHERE reservation_id = ?",
                (reservation_id,),
            )
            conn.executemany(
                "INSERT INTO reservation_participants"
                " (reservation_id, participant_id, start_utc, end_utc)"
                " VALUES (?, ?, ?, ?)",
                [(reservation_id, pid, start_s, end_s) for pid in participants],
            )
            conn.execute("COMMIT")
            return Reservation(
                id=reservation_id,
                request_id=row["request_id"],
                start_utc=_parse(start_s),
                end_utc=_parse(end_s),
                participants=participants,
                created_at=_parse(row["created_at"]),
                version=current_version + 1,
            )
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def cancel_reservation(
    reservation_id: int,
    expected_version: int,
    db_path: str | None = None,
) -> Reservation:
    """取消预约（软删除）：释放全部参与者占用，成功后 ``version`` 加 1。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 预约不存在 → :class:`ReservationNotFoundError`（404）；
    2. 预约仍为 ``active`` 且 ``expected_version`` 与当前版本不一致
       → :class:`VersionConflictError`（409，携带 ``current_version``）；
    3. 预约已取消：
       - ``expected_version == 当前版本 - 1``（即取消成功时所用的版本）
         → **幂等重放**，返回同一取消结果（相同 status/canceled_at/version），
         不重复写入；
       - 其余版本 → :class:`VersionConflictError`（409，携带当前版本）。

    成功时把预约行置为 ``status='canceled'``、写入 ``canceled_at``、
    ``version`` 加 1，并**删除该预约的全部参与者占用**——半开区间立即释放，
    其他请求可立刻再次预约同一时段。``request_id``、``created_at`` 以及
    原时间/参与者快照保持不变。
    """
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：与创建/改期路径一致，事务开始即持写锁，
        # “存在性 + 版本 + 软删除 + 释放占用” 对并发请求原子。
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                f"SELECT {_RESERVATION_COLUMNS} FROM reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise ReservationNotFoundError(reservation_id)

            current_version = row["version"]
            if row["status"] == STATUS_CANCELED:
                # 取消成功会把 version 从 v 推进到 v+1；只有携带
                # “取消时所用的 expected_version == current_version - 1”
                # 的重试才视为同一取消请求的幂等重放。
                if expected_version == current_version - 1:
                    conn.execute("COMMIT")
                    return _row_to_reservation(row)
                raise VersionConflictError(reservation_id, current_version)

            if expected_version != current_version:
                raise VersionConflictError(reservation_id, current_version)

            canceled_s = _canon(datetime.now(timezone.utc))
            conn.execute(
                "UPDATE reservations"
                " SET status = ?, canceled_at = ?, version = version + 1"
                " WHERE id = ?",
                (STATUS_CANCELED, canceled_s, reservation_id),
            )
            # 删除全部参与者占用：取消后该预约不再阻塞任何时段，
            # 参与者可立即在相同半开区间再次被预约。
            conn.execute(
                "DELETE FROM reservation_participants WHERE reservation_id = ?",
                (reservation_id,),
            )
            conn.execute("COMMIT")
            return Reservation(
                id=reservation_id,
                request_id=row["request_id"],
                start_utc=_parse(row["start_utc"]),
                end_utc=_parse(row["end_utc"]),
                # 取消后占用已删除；参与者快照保留在 participants_key 中，
                # 响应回显取消前的参与者集合。
                participants=tuple(json.loads(row["participants_key"])),
                created_at=_parse(row["created_at"]),
                version=current_version + 1,
                status=STATUS_CANCELED,
                canceled_at=_parse(canceled_s),
            )
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()
