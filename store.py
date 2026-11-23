"""SQLite 持久化: 任务/片段/说话人/编辑记录/文档。WAL 模式, 单连接+锁。

设计约束:
- segments.raw_text 是 ASR 不可变原始结果, 任何代码不得改写它。
- corrected_text 默认 NULL; LLM 校对写 corrected_by='llm'; 人工编辑写 corrected_by='human' 并记 edits。
- 说话人显示名放 speakers 表, 改名/合并不动片段行(合并只 UPDATE segment.speaker_id)。
"""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY,
  created_at REAL, updated_at REAL,
  filename TEXT, size_bytes INTEGER,
  status TEXT,               -- queued|processing|done|failed|cancelled
  stage TEXT,                -- uploaded|decoding|asr|done
  error TEXT,
  cancel_requested INTEGER DEFAULT 0,
  hotword TEXT DEFAULT '',
  audio_path TEXT,
  duration_sec REAL,
  deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS segments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT, idx INTEGER,
  start_ms INTEGER, end_ms INTEGER,
  speaker_id INTEGER,        -- NULL = 未知说话人
  raw_text TEXT,
  corrected_text TEXT,
  corrected_by TEXT,         -- NULL|llm|human
  UNIQUE(task_id, idx)
);
CREATE TABLE IF NOT EXISTS speakers(
  task_id TEXT, speaker_id INTEGER, display_name TEXT,
  PRIMARY KEY(task_id, speaker_id)
);
CREATE TABLE IF NOT EXISTS edits(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT, segment_id INTEGER, field TEXT,
  old_value TEXT, new_value TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT, kind TEXT,   -- understanding|transcript|outline|minutes|report
  content TEXT, meta TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS ai_jobs(
  id TEXT PRIMARY KEY,
  task_id TEXT,
  kind TEXT,                 -- proofread|understand|document
  status TEXT,               -- queued|running|done|failed|cancelled
  progress REAL DEFAULT 0,
  detail TEXT DEFAULT '',
  params TEXT DEFAULT '{}',
  result TEXT,
  error TEXT,
  cancel_requested INTEGER DEFAULT 0,
  created_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_segments_task ON segments(task_id, idx);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_jobs_task ON ai_jobs(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON ai_jobs(status);
"""


def _now() -> float:
    return time.time()


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    def __init__(self, db_path: Path):
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------- tasks ----------------
    def create_task(self, filename: str, size: int, hotword: str, audio_path=None) -> str:
        tid = new_task_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks(id, created_at, updated_at, filename, size_bytes,"
                " status, stage, hotword, audio_path) VALUES(?,?,?,?,?,?,?,?,?)",
                (tid, _now(), _now(), filename, size, "queued", "uploaded", hotword, audio_path))
            self._conn.commit()
        return tid

    def find_active_by_file(self, filename: str, size: int) -> str | None:
        """同文件(名+大小)已有排队/处理中任务 → 返回其 id, 避免重复处理。"""
        with self._lock:
            r = self._conn.execute(
                "SELECT id FROM tasks WHERE deleted=0 AND filename=? AND size_bytes=?"
                " AND status IN ('queued','processing') LIMIT 1",
                (filename, size)).fetchone()
        return r["id"] if r else None

    def update_task(self, tid: str, **kw) -> None:
        kw["updated_at"] = _now()
        cols = ",".join(f"{k}=?" for k in kw)
        with self._lock:
            self._conn.execute(f"UPDATE tasks SET {cols} WHERE id=?", (*kw.values(), tid))
            self._conn.commit()

    def get_task(self, tid: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        return dict(r) if r else None

    def list_tasks(self, include_deleted=False) -> list[dict]:
        sql = "SELECT * FROM tasks"
        if not include_deleted:
            sql += " WHERE deleted=0"
        sql += " ORDER BY created_at DESC LIMIT 500"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql).fetchall()]

    def recover_on_startup(self) -> dict:
        """服务重启恢复: processing 标记为 failed(可重试); queued 原样保留由 worker 重新入队。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM tasks WHERE status='processing' AND deleted=0").fetchall()
            self._conn.execute(
                "UPDATE tasks SET status='failed', error='服务重启中断, 可重试', updated_at=?"
                " WHERE status='processing' AND deleted=0", (_now(),))
            self._conn.commit()
        return {"interrupted": [r["id"] for r in cur]}

    def pending_queued(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM tasks WHERE status='queued' AND deleted=0 ORDER BY created_at").fetchall()
        return [r["id"] for r in rows]

    # ---------------- AI jobs ----------------
    @staticmethod
    def _params_json(params: dict | None) -> str:
        return json.dumps(params or {}, ensure_ascii=False, sort_keys=True)

    def create_job(self, task_id: str, kind: str, params: dict | None = None) -> str:
        jid = new_task_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO ai_jobs(id, task_id, kind, status, params, created_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (jid, task_id, kind, "queued", self._params_json(params), _now(), _now()))
            self._conn.commit()
        return jid

    def get_job(self, jid: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM ai_jobs WHERE id=?", (jid,)).fetchone()
        return dict(r) if r else None

    def update_job(self, jid: str, **kw) -> None:
        kw["updated_at"] = _now()
        cols = ",".join(f"{k}=?" for k in kw)
        with self._lock:
            self._conn.execute(f"UPDATE ai_jobs SET {cols} WHERE id=?", (*kw.values(), jid))
            self._conn.commit()

    def list_jobs(self, task_id: str, kind: str | None = None,
                  active_only: bool = False, limit: int = 20) -> list[dict]:
        sql = "SELECT * FROM ai_jobs WHERE task_id=?"
        args: list = [task_id]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if active_only:
            sql += " AND status IN ('queued','running')"
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, int(limit)))
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def find_active_job(self, task_id: str, kind: str, params: dict | None = None) -> str | None:
        """同 kind(可选同参数)已有排队/运行中任务 → 返回 job id, 供幂等去重。"""
        sql = "SELECT id FROM ai_jobs WHERE task_id=? AND kind=?" \
              " AND status IN ('queued','running')"
        args: list = [task_id, kind]
        if params is not None:
            sql += " AND params=?"
            args.append(self._params_json(params))
        sql += " LIMIT 1"
        with self._lock:
            r = self._conn.execute(sql, args).fetchone()
        return r["id"] if r else None

    def has_active_job(self, task_id: str) -> bool:
        with self._lock:
            r = self._conn.execute(
                "SELECT 1 FROM ai_jobs WHERE task_id=? AND status IN ('queued','running') LIMIT 1",
                (task_id,)).fetchone()
        return r is not None

    def recover_jobs(self) -> dict:
        """服务重启: running 标记 failed(可重新发起); queued 由 AIWorker 重新入队。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM ai_jobs WHERE status='running'").fetchall()
            self._conn.execute(
                "UPDATE ai_jobs SET status='failed', error='服务重启中断, 可重新发起',"
                " detail='中断', updated_at=? WHERE status='running'", (_now(),))
            self._conn.commit()
        return {"interrupted": [r["id"] for r in rows]}

    def pending_queued_jobs(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM ai_jobs WHERE status='queued' ORDER BY created_at").fetchall()
        return [r["id"] for r in rows]

    # ---------------- segments / speakers ----------------
    def replace_segments(self, tid: str, segments: list[dict]) -> None:
        """写入 ASR 原始结果(整批替换, 幂等)。dict: idx,start_ms,end_ms,speaker_id,raw_text"""
        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE task_id=?", (tid,))
            self._conn.executemany(
                "INSERT INTO segments(task_id, idx, start_ms, end_ms, speaker_id, raw_text)"
                " VALUES(?,?,?,?,?,?)",
                [(tid, s["idx"], s.get("start_ms"), s.get("end_ms"),
                  s.get("speaker_id"), s.get("raw_text") or "") for s in segments])
            self._conn.commit()

    def get_segments(self, tid: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE task_id=? ORDER BY idx", (tid,)).fetchall()
        return [dict(r) for r in rows]

    def update_segment(self, seg_id: int, task_id: str, field: str, new,
                       corrected_by: str) -> bool:
        """人工单段修改。old 从库里取当前值(不信客户端); 改说话人不影响文本修正来源。"""
        if field not in ("text", "speaker_id"):
            raise ValueError(f"unknown field {field}")
        with self._lock:
            r = self._conn.execute(
                "SELECT raw_text, corrected_text, speaker_id FROM segments"
                " WHERE id=? AND task_id=?", (seg_id, task_id)).fetchone()
            if not r:
                return False
            if field == "text":
                old = r["corrected_text"] if r["corrected_text"] is not None else r["raw_text"]
                self._conn.execute(
                    "UPDATE segments SET corrected_text=?, corrected_by=? WHERE id=?",
                    (new, corrected_by, seg_id))
            else:
                old = r["speaker_id"]
                value = int(new) if new is not None else None
                self._conn.execute(
                    "UPDATE segments SET speaker_id=? WHERE id=?", (value, seg_id))
            self._conn.execute(
                "INSERT INTO edits(task_id, segment_id, field, old_value, new_value, ts)"
                " VALUES(?,?,?,?,?,?)",
                (task_id, seg_id, field,
                 None if old is None else str(old),
                 None if new is None else str(new), _now()))
            self._conn.commit()
        return True

    def apply_corrections(self, tid: str, mapping: dict[int, str], by: str = "llm",
                          skip_human: bool = True) -> int:
        """批量回填校对文本(按 idx)。默认绝不覆盖人工修改; 返回实际写入段数。"""
        if not mapping:
            return 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, idx, raw_text, corrected_text, corrected_by FROM segments"
                " WHERE task_id=?", (tid,)).fetchall()
            n = 0
            for r in rows:
                new = mapping.get(r["idx"])
                if new is None:
                    continue
                if skip_human and r["corrected_by"] == "human":
                    continue
                old = r["corrected_text"] if r["corrected_text"] is not None else r["raw_text"]
                self._conn.execute(
                    "UPDATE segments SET corrected_text=?, corrected_by=? WHERE id=?",
                    (new, by, r["id"]))
                if old != new:
                    self._conn.execute(
                        "INSERT INTO edits(task_id, segment_id, field, old_value, new_value, ts)"
                        " VALUES(?,?,?,?,?,?)",
                        (tid, r["id"], "text", old, new, _now()))
                n += 1
            self._conn.commit()
        return n

    def count_missing_corrections(self, tid: str) -> int:
        """尚未被 LLM/人工修正过的片段数(补缺入口用)。"""
        with self._lock:
            r = self._conn.execute(
                "SELECT COUNT(*) FROM segments WHERE task_id=? AND corrected_text IS NULL",
                (tid,)).fetchone()
        return int(r[0])

    def get_speakers(self, tid: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM speakers WHERE task_id=? ORDER BY speaker_id", (tid,)).fetchall()
        return [dict(r) for r in rows]

    def upsert_speaker(self, tid: str, speaker_id: int, display_name: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO speakers(task_id, speaker_id, display_name) VALUES(?,?,?)"
                " ON CONFLICT(task_id, speaker_id) DO UPDATE SET display_name=excluded.display_name",
                (tid, speaker_id, display_name))
            self._conn.commit()

    def reassign_speakers(self, tid: str, seg_ids: list[int], to_speaker: int | None,
                          new_speaker_id: int | None = None) -> int:
        """把片段改派给已有说话人(to_speaker)或新说话人(new_speaker_id, 用于'拆分')。
        返回改派数量。"""
        target = new_speaker_id if to_speaker is None else to_speaker
        if not seg_ids:
            return 0
        marks = ",".join("?" * len(seg_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, speaker_id FROM segments WHERE task_id=? AND id IN ({marks})",
                (tid, *seg_ids)).fetchall()
            changed = 0
            for r in rows:
                if r["speaker_id"] == target:
                    continue
                self._conn.execute(
                    "INSERT INTO edits(task_id, segment_id, field, old_value, new_value, ts)"
                    " VALUES(?,?,?,?,?,?)",
                    (tid, r["id"], "speaker_id", str(r["speaker_id"]), str(target), _now()))
                changed += 1
            self._conn.execute(
                f"UPDATE segments SET speaker_id=? WHERE task_id=? AND id IN ({marks})",
                (target, tid, *seg_ids))
            if new_speaker_id is not None:
                self._conn.execute(
                    "INSERT OR IGNORE INTO speakers(task_id, speaker_id, display_name)"
                    " VALUES(?,?,?)", (tid, new_speaker_id, f"说话人{new_speaker_id + 1}"))
            self._conn.commit()
            return changed

    def merge_speakers(self, tid: str, from_ids: list[int], to_id: int) -> int:
        """合并说话人标签: from_ids 的所有片段划入 to_id, 并删除 from 名牌(写编辑历史)。"""
        if not from_ids:
            return 0
        marks = ",".join("?" * len(from_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, speaker_id FROM segments WHERE task_id=? AND speaker_id IN ({marks})",
                (tid, *from_ids)).fetchall()
            for r in rows:
                self._conn.execute(
                    "INSERT INTO edits(task_id, segment_id, field, old_value, new_value, ts)"
                    " VALUES(?,?,?,?,?,?)",
                    (tid, r["id"], "speaker_id", str(r["speaker_id"]), str(to_id), _now()))
            self._conn.execute(
                f"UPDATE segments SET speaker_id=? WHERE task_id=? AND speaker_id IN ({marks})",
                (to_id, tid, *from_ids))
            self._conn.execute(
                f"DELETE FROM speakers WHERE task_id=? AND speaker_id IN ({marks})",
                (tid, *from_ids))
            self._conn.commit()
            return len(rows)

    def get_edits(self, tid: str, limit=200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM edits WHERE task_id=? ORDER BY ts DESC LIMIT ?",
                (tid, limit)).fetchall()
        return [dict(r) for r in rows]

    # ---------------- documents ----------------
    def put_document(self, tid: str, kind: str, content: str, meta: dict | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO documents(task_id, kind, content, meta, created_at) VALUES(?,?,?,?,?)",
                (tid, kind, content, json.dumps(meta or {}, ensure_ascii=False), _now()))
            self._conn.commit()
            return cur.lastrowid

    def get_documents(self, tid: str, kind: str | None = None) -> list[dict]:
        if kind:
            sql = "SELECT * FROM documents WHERE task_id=? AND kind=? ORDER BY id DESC"
            args = (tid, kind)
        else:
            sql = "SELECT * FROM documents WHERE task_id=? ORDER BY id DESC"
            args = (tid,)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # ---------------- cleanup ----------------
    def purge_task(self, tid: str) -> dict:
        """删除任务的全部内容(片段/说话人/编辑/文档/AI 任务), 保留一条脱敏墓碑行。"""
        with self._lock:
            r = self._conn.execute("SELECT audio_path FROM tasks WHERE id=?", (tid,)).fetchone()
            for table in ("segments", "speakers", "edits", "documents", "ai_jobs"):
                self._conn.execute(f"DELETE FROM {table} WHERE task_id=?", (tid,))
            self._conn.execute(
                "UPDATE tasks SET deleted=1, audio_path=NULL, hotword='', filename='(已删除)',"
                " error=NULL, cancel_requested=0, updated_at=? WHERE id=?", (_now(), tid))
            self._conn.commit()
        return {"audio_path": r["audio_path"] if r else None}
