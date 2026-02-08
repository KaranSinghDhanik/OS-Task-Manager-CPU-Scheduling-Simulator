import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import psutil


@dataclass
class RuntimeRecord:
	pid: int
	name: str
	create_time: float
	predicted: float
	first_seen: float
	last_seen: float
	actual_duration: Optional[float] = None
	completed_at: Optional[float] = None

	def elapsed(self, now: Optional[float] = None) -> float:
		if self.actual_duration is not None:
			return self.actual_duration
		now = now or time.time()
		return max(now - self.create_time, 0.0)

	@property
	def status(self) -> str:
		return "completed" if self.actual_duration is not None else "running"


class ProcessRuntimeTracker:
	def __init__(self, completed_max: int = 20) -> None:
		self._active: Dict[int, RuntimeRecord] = {}
		self._completed: Deque[RuntimeRecord] = deque(maxlen=completed_max)

	def _replace_record(self, pid: int, name: str, create_time: float, predicted: float, now: float) -> RuntimeRecord:
		record = RuntimeRecord(
			pid=pid,
			name=name,
			create_time=create_time,
			predicted=float(predicted),
			first_seen=now,
			last_seen=now,
		)
		self._active[pid] = record
		return record

	def update_running(self, pid: int, name: str, create_time: float, predicted: float) -> RuntimeRecord:
		now = time.time()
		record = self._active.get(pid)
		if record is None:
			return self._replace_record(pid, name, create_time, predicted, now)

		# If PID was reused for a new process, reset the record.
		if abs(record.create_time - create_time) > 1.0:
			return self._replace_record(pid, name, create_time, predicted, now)

		record.name = name
		record.predicted = float(predicted)
		record.last_seen = now
		return record

	def mark_missing(self, active_pids: List[int]) -> None:
		active_set = set(active_pids)
		now = time.time()
		for pid in list(self._active.keys()):
			if pid in active_set:
				continue

			record = self._active.get(pid)
			if record is None:
				continue

			should_close = False
			try:
				proc = psutil.Process(pid)
				with proc.oneshot():
					create_time = proc.create_time()
				if abs(create_time - record.create_time) > 1.0:
					should_close = True
				else:
					# Still same process; keep tracking.
					continue
			except psutil.NoSuchProcess:
				should_close = True
			except psutil.AccessDenied:
				# If we can't access, assume still running to avoid false completion.
				continue

			if should_close:
				record = self._active.pop(pid, None)
				if record is None:
					continue
				record.completed_at = now
				record.actual_duration = max(record.completed_at - record.create_time, 0.0)
				self._completed.append(record)

	def current_elapsed(self, pid: int) -> Optional[float]:
		record = self._active.get(pid)
		if record:
			return record.elapsed()
		for record in self._completed:
			if record.pid == pid:
				return record.actual_duration
		return None

	def status_for(self, pid: int) -> str:
		record = self._active.get(pid)
		if record:
			return record.status
		for record in self._completed:
			if record.pid == pid:
				return record.status
		return "unknown"

	def recent_completions(self) -> List[RuntimeRecord]:
		return list(self._completed)

