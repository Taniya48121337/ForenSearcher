"""邀请试用的凭证、会话与撤销。"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SESSION_SECONDS = 8 * 60 * 60
COOKIE_NAME = "foren_session"


def _digest(token: str) -> str:
    # 随机 256 位凭证只保存摘要；数据库泄露时不能直接拿记录登录。
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TrialAccess:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    invite_digest TEXT NOT NULL UNIQUE,
                    expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_digest TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS login_attempts (client TEXT NOT NULL, attempted_at REAL NOT NULL);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def invite(self, name: str, days: int = 7) -> dict[str, Any]:
        if not name.strip() or len(name) > 80 or not 1 <= days <= 90:
            raise ValueError("名称需为1至80字，有效期需为1至90天")
        token, identifier = secrets.token_urlsafe(32), uuid.uuid4().hex
        expires = time.time() + days * 86400
        with self.connect() as db:
            db.execute("INSERT INTO users (id,name,invite_digest,expires_at) VALUES (?,?,?,?)",
                       (identifier, name.strip(), _digest(token), expires))
        return {"id": identifier, "name": name.strip(), "invite_code": token, "expires_at": expires}

    def revoke(self, identifier: str) -> bool:
        with self.connect() as db:
            changed = db.execute("UPDATE users SET revoked=1 WHERE id=?", (identifier,)).rowcount
            db.execute("DELETE FROM sessions WHERE user_id=?", (identifier,))
        return bool(changed)

    def login(self, token: str, client: str) -> tuple[str, dict[str, Any]] | None:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM login_attempts WHERE attempted_at < ?", (now - 60,))
            counts = db.execute("SELECT count(*) AS total, sum(client=?) AS same FROM login_attempts", (client,)).fetchone()
            if counts["total"] >= 60 or (counts["same"] or 0) >= 10:
                raise PermissionError("登录尝试过于频繁，请一分钟后重试")
            db.execute("INSERT INTO login_attempts VALUES (?,?)", (client, now))
            user = db.execute("SELECT id,name,expires_at FROM users WHERE invite_digest=? AND revoked=0 AND expires_at>?",
                              (_digest(token), now)).fetchone() if len(token) == 43 else None
            if user is None:
                return None
            session = secrets.token_urlsafe(32)
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (now,))
            db.execute("INSERT INTO sessions VALUES (?,?,?)", (_digest(session), user["id"], min(now + SESSION_SECONDS, user["expires_at"])))
            return session, {"id": user["id"], "name": user["name"]}

    def identity(self, token: str) -> dict[str, str] | None:
        if len(token) != 43:
            return None
        now = time.time()
        with self.connect() as db:
            user = db.execute("""SELECT u.id,u.name FROM sessions s JOIN users u ON u.id=s.user_id
                WHERE s.token_digest=? AND s.expires_at>? AND u.expires_at>? AND u.revoked=0""",
                              (_digest(token), now, now)).fetchone()
        return {"id": user["id"], "name": user["name"]} if user else None

    def logout(self, token: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE token_digest=?", (_digest(token),))
