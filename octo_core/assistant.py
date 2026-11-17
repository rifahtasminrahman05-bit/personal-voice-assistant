# -*- coding: utf-8 -*-
"""Personal Assistant (features 69-82).

Calendar, reminders, alarms, tasks, email drafting, meeting scheduling,
daily briefing, offline weather cache, news summarization, notes,
contacts, travel planning, time management and goal tracking.
100% local: JSON persistence, no network, no external APIs.
"""
import json
import os
from datetime import datetime, timedelta

from octo.summarize import summarize


def _now():
    return datetime.now().isoformat(timespec="seconds")


class PersonalAssistant:
    """Feature 69-82: local personal assistant with JSON persistence."""

    PRIORITIES = {"high": 0, "medium": 1, "low": 2}

    def __init__(self, store="data/octo/assistant.json"):
        self.path = store
        os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
        self.data = {
            "events": [], "reminders": [], "alarms": [], "tasks": [],
            "email_outbox": [], "meetings": [], "notes": [],
            "contacts": [], "weather_cache": {}, "goals": [],
        }
        self._load()

    # ---- persistence ----
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    stored = json.load(fh)
                for key, value in stored.items():
                    if key in self.data:
                        self.data[key] = value
            except (OSError, ValueError):
                pass

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=1)

    # ---- 69. calendar ----
    def calendar_add(self, title, when):
        event = {"title": title, "when": when, "created": _now()}
        self.data["events"].append(event)
        self._save()
        return event

    def calendar_upcoming(self, limit=5):
        events = sorted(self.data["events"], key=lambda e: e["when"])
        return events[:limit]

    # ---- 70. reminders ----
    def remind(self, text, when):
        item = {"text": text, "when": when, "done": False}
        self.data["reminders"].append(item)
        self._save()
        return item

    def due_reminders(self, now=None):
        now = now or _now()
        due = [r for r in self.data["reminders"]
               if not r["done"] and r["when"] <= now]
        for item in due:
            item["done"] = True
        if due:
            self._save()
        return due

    # ---- 71. alarms ----
    def alarm_set(self, time_str, label="alarm"):
        alarm = {"time": time_str, "label": label, "on": True}
        self.data["alarms"].append(alarm)
        self._save()
        return alarm

    def alarm_off(self, label):
        for alarm in self.data["alarms"]:
            if alarm["label"] == label:
                alarm["on"] = False
        self._save()

    # ---- 72. tasks ----
    def task_add(self, text, priority="medium"):
        task = {"text": text, "priority": priority,
                "done": False, "created": _now()}
        self.data["tasks"].append(task)
        self._save()
        return task

    def task_complete(self, text):
        for task in self.data["tasks"]:
            if task["text"] == text and not task["done"]:
                task["done"] = True
                self._save()
                return True
        return False

    def task_list(self):
        return sorted(self.data["tasks"],
                      key=lambda t: (t["done"],
                                     self.PRIORITIES.get(t["priority"], 1)))

    # ---- 73. email (drafts only, never sent) ----
    def email_draft(self, to, subject, body):
        mail = {"to": to, "subject": subject, "body": body,
                "status": "draft", "created": _now()}
        self.data["email_outbox"].append(mail)
        self._save()
        return mail

    def email_outbox(self):
        return list(self.data["email_outbox"])

    # ---- 74. meetings ----
    def schedule_meeting(self, title, when, attendees=()):
        conflict = any(m["when"] == when for m in self.data["meetings"])
        meeting = {"title": title, "when": when,
                   "attendees": list(attendees), "conflict": conflict}
        self.data["meetings"].append(meeting)
        self._save()
        return meeting

    # ---- 75. daily briefing ----
    def daily_briefing(self, today=None):
        today = today or datetime.now().strftime("%Y-%m-%d")
        events = [e for e in self.data["events"]
                  if str(e["when"]).startswith(today)]
        meetings = [m for m in self.data["meetings"]
                    if str(m["when"]).startswith(today)]
        tasks = [t for t in self.task_list() if not t["done"]][:5]
        reminders = [r for r in self.data["reminders"] if not r["done"]]
        return {"date": today, "events": events, "meetings": meetings,
                "top_tasks": tasks, "open_reminders": len(reminders)}

    # ---- 76. weather (OFFLINE: local cache only) ----
    def weather(self, city, condition=None, temp_c=None):
        if condition is not None:
            entry = {"condition": condition, "temp_c": temp_c,
                     "updated": _now()}
            self.data["weather_cache"][city] = entry
            self._save()
            return {"ok": True, "city": city, **entry}
        cached = self.data["weather_cache"].get(city)
        if cached is None:
            return {"ok": False, "city": city,
                    "error": "no cached weather (offline mode)"}
        return {"ok": True, "city": city, **cached}

    # ---- 77. news summarization (local text only) ----
    def news_digest(self, text, n=3):
        return {"summary": summarize(text, n=n)}

    # ---- 78. notes ----
    def note_add(self, text, tags=()):
        note = {"text": text, "tags": list(tags), "created": _now()}
        self.data["notes"].append(note)
        self._save()
        return note

    def note_search(self, query):
        query = query.lower()
        return [n for n in self.data["notes"]
                if query in n["text"].lower()
                or any(query in t.lower() for t in n["tags"])]

    # ---- 79. contacts ----
    def contact_add(self, name, phone=None, email=None):
        contact = {"name": name, "phone": phone, "email": email}
        self.data["contacts"].append(contact)
        self._save()
        return contact

    def contact_find(self, name):
        name = name.lower()
        return [c for c in self.data["contacts"]
                if name in c["name"].lower()]

    # ---- 80. travel planning ----
    def travel_plan(self, destination, days, interests=()):
        interests = list(interests) or ["sightseeing", "food", "rest"]
        itinerary = []
        for day in range(days):
            focus = interests[day % len(interests)]
            itinerary.append({"day": day + 1, "destination": destination,
                              "focus": focus})
        plan = {"destination": destination, "days": days,
                "itinerary": itinerary}
        self.data.setdefault("travels", []).append(plan)
        self._save()
        return plan

    # ---- 81. time management ----
    def time_plan(self, blocks):
        """blocks: [{"label", "hours"}] -> normalized shares."""
        total = sum(b.get("hours", 0) for b in blocks) or 1
        plan = [{"label": b["label"],
                 "share": round(b.get("hours", 0) / total, 3)}
                for b in blocks]
        return {"blocks": plan, "total_hours": total}

    # ---- 82. goal tracking ----
    def goal_add(self, text, target=None):
        goal = {"text": text, "target": target, "progress": 0.0,
                "created": _now()}
        self.data["goals"].append(goal)
        self._save()
        return goal

    def goal_progress(self, text, progress):
        for goal in self.data["goals"]:
            if goal["text"] == text:
                goal["progress"] = max(0.0, min(1.0, float(progress)))
                self._save()
                return goal
        return None
