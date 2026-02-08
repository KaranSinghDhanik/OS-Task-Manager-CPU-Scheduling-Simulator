import psutil
import time
from typing import Dict, Optional, List
import statistics


def list_processes():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time']):
        try:
            cpu_times = proc.cpu_times()
            total_cpu_time = cpu_times.user + cpu_times.system
            processes.append({
                "pid": proc.info['pid'],
                "name": proc.info['name'],
                "cpu_percent": proc.info['cpu_percent'],
                "memory_percent": proc.info['memory_percent'],
                "cpu_time": total_cpu_time,
                "create_time": proc.info['create_time']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def kill_process(pid: int):
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        return {"status": "success", "message": f"Process {pid} terminated."}
    except psutil.NoSuchProcess:
        return {"status": "error", "message": "Process not found."}
    except psutil.AccessDenied:
        return {"status": "error", "message": "Permission denied."}


# ---------------------------------
# Feature Extraction from Live System
# ---------------------------------

def compute_rr_features_from_system(time_quantum: float = 1.0) -> Dict[str, float]:
    """
    Derive model input features from current system processes.
    - mean_burst/std_burst: stats of per-process CPU time (user+system)
    - mean_arrival/std_arrival: stats of process ages (now - create_time)
    - num_processes: count of sampled processes
    - system_load: overall CPU utilization (0..1)
    - time_quantum: provided parameter
    """
    procs = list_processes()
    now = time.time()

    cpu_times = [p["cpu_time"] for p in procs if p.get("cpu_time") is not None]
    ages = [max(now - p["create_time"], 0) for p in procs if p.get("create_time") is not None]

    def safe_mean(values):
        return float(statistics.mean(values)) if values else 0.0

    def safe_stdev(values):
        return float(statistics.pstdev(values)) if values else 0.0

    mean_burst = safe_mean(cpu_times)
    std_burst = safe_stdev(cpu_times)

    mean_arrival = safe_mean(ages)
    std_arrival = safe_stdev(ages)

    num_processes = len(procs)
    # Use system-wide CPU percent over a short interval; divide by 100 to map to 0..1
    system_load = psutil.cpu_percent(interval=0.1) / 100.0

    return {
        "mean_burst": mean_burst,
        "std_burst": std_burst,
        "mean_arrival": mean_arrival,
        "std_arrival": std_arrival,
        "num_processes": float(num_processes),
        "system_load": system_load,
        "time_quantum": float(time_quantum),
    }


def _get_process_snapshot(pid: int) -> Optional[Dict]:
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            cpu_times = proc.cpu_times()
            total_cpu_time = cpu_times.user + cpu_times.system
            return {
                "pid": proc.pid,
                "name": proc.name(),
                "cpu_time": total_cpu_time,
                "create_time": proc.create_time()
            }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def get_process_snapshot(pid: int) -> Optional[Dict]:
	return _get_process_snapshot(pid)


def compute_rr_features_for_pid(pid: int, time_quantum: float = 1.0) -> Optional[Dict[str, float]]:
    """
    Derive model input features for a single process.
    Maps burst to that process's CPU time; arrival to its age.
    """
    snap = _get_process_snapshot(pid)
    if snap is None:
        return None

    now = time.time()
    cpu_time = float(snap["cpu_time"])
    age = float(max(now - snap["create_time"], 0))

    system_load = psutil.cpu_percent(interval=0.05) / 100.0

    return {
        "mean_burst": cpu_time,
        "std_burst": 0.0,
        "mean_arrival": age,
        "std_arrival": 0.0,
        "num_processes": 1.0,
        "system_load": system_load,
        "time_quantum": float(time_quantum),
    }


def pick_top_processes(top_n: int = 10, sort_by: str = "cpu_percent", order: str = "desc") -> List[Dict]:
    procs = list_processes()
    reverse = (order == "desc")
    try:
        procs.sort(key=lambda p: (p.get(sort_by) is None, p.get(sort_by)), reverse=reverse)
    except Exception:
        pass
    return procs[:max(0, top_n)]
