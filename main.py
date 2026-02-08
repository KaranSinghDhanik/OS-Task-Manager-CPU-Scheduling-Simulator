from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

from process_manager import list_processes, kill_process, compute_rr_features_from_system, compute_rr_features_for_pid, pick_top_processes, get_process_snapshot
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from model.predict import predict_completion
from model.train import train as train_model
from runtime_tracker import ProcessRuntimeTracker

app = FastAPI()

runtime_tracker = ProcessRuntimeTracker()


def _runtime_metadata(pid: int, name: str, create_time: Optional[float], predicted: float):
	if create_time is None:
		return None, "unknown"

	record = runtime_tracker.update_running(pid, name, float(create_time), predicted)
	actual = runtime_tracker.current_elapsed(pid)
	status = runtime_tracker.status_for(pid)
	return (float(actual) if actual is not None else None), status


def _serialize_completion(record):
	return {
		"pid": record.pid,
		"name": record.name,
		"predicted_turnaround_time": float(record.predicted),
		"actual_turnaround_time": float(record.actual_duration) if record.actual_duration is not None else None,
		"turnaround_status": record.status,
		"completed_at": float(record.completed_at) if record.completed_at is not None else None,
		"duration_error": (float(record.actual_duration) - float(record.predicted)) if record.actual_duration is not None else None,
	}

# Allow cross-origin requests for local development and future frontend
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # TODO: restrict in production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Serve simple static frontend (mounted at /app)
app.mount("/app", StaticFiles(directory="../task-manager-frontend", html=True), name="app")

@app.get("/", include_in_schema=False)
def root_redirect():
	return RedirectResponse(url="/app/")
class PredictRRResponse(BaseModel):
	predicted_turnaround_time: float
	features: Dict[str, Any]

class PredictProcessResponse(BaseModel):
	pid: int
	name: str
	predicted_turnaround_time: float
	actual_turnaround_time: Optional[float] = Field(default=None)
	turnaround_status: Literal["running", "completed", "unknown"] = "unknown"
	features: Dict[str, Any]

class PredictProcessesResponse(BaseModel):
	count: int
	items: List[PredictProcessResponse]
	params: Dict[str, Any]


@app.get("/processes")
def get_processes(
	page: int = Query(1, gt=0),
	page_size: int = Query(25, gt=0, le=500),
	sort_by: Literal["name", "cpu_percent", "memory_percent", "cpu_time", "create_time"] = "cpu_percent",
	order: Literal["asc", "desc"] = "desc"
):
	procs = list_processes()

	# Sorting
	reverse = (order == "desc")
	try:
		procs.sort(key=lambda p: (p.get(sort_by) is None, p.get(sort_by)), reverse=reverse)
	except Exception:
		# Fallback to no sort if unexpected data
		pass

	# Pagination
	total = len(procs)
	start = (page - 1) * page_size
	end = start + page_size
	data = procs[start:end]

	return {
		"page": page,
		"page_size": page_size,
		"total": total,
		"results": data
	}

@app.post("/kill/{pid}")
def terminate_process(pid: int):
    return kill_process(pid)

@app.get("/predict/system", response_model=PredictRRResponse)
def predict_from_system(quantum: float = Query(1.0, gt=0)):
	"""
	Compute model features from live system processes and predict turnaround time.
	"""
	try:
		features = compute_rr_features_from_system(quantum)
		pred = predict_completion(features)
		return {
			"predicted_turnaround_time": pred,
			"features": features
		}
	except FileNotFoundError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		# Return the actual error to aid debugging
		raise HTTPException(status_code=500, detail=f"System prediction failed: {e}")


@app.get("/predict/process/{pid}", response_model=PredictProcessResponse)
def predict_for_process(pid: int, quantum: float = Query(1.0, gt=0)):
	try:
		features = compute_rr_features_for_pid(pid, quantum)
		if features is None:
			raise HTTPException(status_code=404, detail="Process not found or inaccessible")
		pred = predict_completion(features)

		snapshot = get_process_snapshot(pid)
		if snapshot is None:
			raise HTTPException(status_code=404, detail="Process not found or inaccessible")

		name = snapshot.get("name", str(pid))
		create_time = snapshot.get("create_time")
		actual, status = _runtime_metadata(pid, name, create_time, pred)

		return {
			"pid": pid,
			"name": name,
			"predicted_turnaround_time": pred,
			"actual_turnaround_time": actual,
			"turnaround_status": status,
			"features": features
		}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Per-process prediction failed: {e}")


@app.get("/predict/processes", response_model=PredictProcessesResponse)
def predict_for_top_processes(
	top_n: int = Query(10, gt=0, le=100),
	sort_by: Literal["name","cpu_percent","memory_percent","cpu_time","create_time"] = "cpu_percent",
	order: Literal["asc","desc"] = "desc",
	quantum: float = Query(1.0, gt=0),
):
	try:
		selected = pick_top_processes(top_n=top_n, sort_by=sort_by, order=order)
		items: List[Dict[str, Any]] = []
		active_pids: List[int] = []
		for p in selected:
			features = compute_rr_features_for_pid(p["pid"], quantum)
			if features is None:
				continue
			pred = predict_completion(features)
			create_time = p.get("create_time")
			actual, status = _runtime_metadata(p["pid"], p["name"], create_time, pred)
			active_pids.append(p["pid"])
			items.append({
				"pid": p["pid"],
				"name": p["name"],
				"predicted_turnaround_time": pred,
				"actual_turnaround_time": actual,
				"turnaround_status": status,
				"features": features
			})

		runtime_tracker.mark_missing(active_pids)

		return {
			"count": len(items),
			"items": items,
			"params": {"top_n": top_n, "sort_by": sort_by, "order": order, "quantum": quantum}
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Batch process prediction failed: {e}")


@app.websocket("/ws")
async def ws_live(websocket: WebSocket):
	# Optional query params: top_n, quantum
	query = websocket.query_params
	try:
		top_n = int(query.get("top_n", 15))
		quantum = float(query.get("quantum", 3.0))
	except Exception:
		top_n = 15
		quantum = 3.0

	await websocket.accept()
	try:
		while True:
			# Build payload: system prediction + per-process items (top N by CPU%)
			features_sys = compute_rr_features_from_system(quantum)
			pred_sys = predict_completion(features_sys)

			selected = pick_top_processes(top_n=top_n, sort_by="cpu_percent", order="desc")
			items = []
			active_pids: List[int] = []
			for p in selected:
				try:
					f = compute_rr_features_for_pid(p["pid"], quantum)
					if f is None:
						continue
					pred = predict_completion(f)
					create_time = p.get("create_time")
					actual, status = _runtime_metadata(p["pid"], p["name"], create_time, pred)
					active_pids.append(p["pid"])
					items.append({
						"pid": p["pid"],
						"name": p["name"],
						"cpu_percent": p.get("cpu_percent", 0),
						"memory_percent": p.get("memory_percent", 0),
						"cpu_time": p.get("cpu_time", 0),
						"predicted_turnaround_time": pred,
						"actual_turnaround_time": actual,
						"turnaround_status": status,
					})
				except Exception:
					# Skip processes that vanished or errored mid-loop
					continue

			runtime_tracker.mark_missing(active_pids)
			recent = [_serialize_completion(r) for r in runtime_tracker.recent_completions()]

			await websocket.send_json({
				"system_prediction": float(pred_sys),
				"items": items,
				"quantum": quantum,
				"count": len(items),
				"recent_completions": recent,
			})
			await asyncio.sleep(2.0)
	except WebSocketDisconnect:
		return
	except Exception:
		# Close socket on unexpected error
		try:
			await websocket.close()
		except Exception:
			pass

@app.post("/train")
def train():
	"""
	Trigger model training (dev-only).
	Returns training metrics and artifact paths.
	"""
	try:
		result = train_model()
		return result
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Training failed: {e}")
