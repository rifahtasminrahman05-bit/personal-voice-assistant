# -*- coding: utf-8 -*-
"""OfflineStorage — local SQLite memory for notes, tasks and reminders."""
import os
import sqlite3
from datetime import datetime

from core import config

DB_PATH = os.path.join(config.BASE_DIR, "data", "memory.db")


class OfflineStorage:
    def __init__(self, path=DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created TEXT NOT NULL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0,
            created TEXT NOT NULL)""")
        self.conn.commit()

    def remember(self, text):
        with self.conn:
            self.conn.execute("INSERT INTO notes (text, created) VALUES (?, ?)",
                              (text, datetime.now().isoformat(timespec="seconds")))
        return f"মনে রাখলাম: {text}"

    def recall(self, query="", limit=5):
        cur = self.conn.execute(
            "SELECT text, created FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{query}%", limit))
        rows = cur.fetchall()
        return rows

    def forget(self, note_id):
        with self.conn:
            cur = self.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cur.rowcount > 0

    def add_task(self, text):
        with self.conn:
            self.conn.execute("INSERT INTO tasks (text, created) VALUES (?, ?)",
                              (text, datetime.now().isoformat(timespec="seconds")))
        return f"টাস্ক যোগ হলো: {text}"

    def list_tasks(self, include_done=False):
        sql = "SELECT id, text, done FROM tasks"
        if not include_done:
            sql += " WHERE done = 0"
        return self.conn.execute(sql + " ORDER BY id").fetchall()

    def complete_task(self, task_id):
        with self.conn:
            cur = self.conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
        return cur.rowcount > 0
