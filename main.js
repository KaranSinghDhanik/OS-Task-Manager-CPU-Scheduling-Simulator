(() => {
	const tbody = document.getElementById("tbody");
	const statusEl = document.getElementById("status");
	const sysPredEl = document.getElementById("systemPrediction");
	const searchInput = document.getElementById("searchInput");

	let lastPayload = null;
	let filterTerm = "";

	function fmt(n, digits = 2) {
		if (n === null || n === undefined || Number.isNaN(n)) return "—";
		return Number(n).toFixed(digits);
	}

	async function kill(pid) {
		try {
			await fetch(`/kill/${pid}`, { method: "POST" });
		} catch (e) {
			console.error("Kill failed", e);
		}
	}

	function matchesFilter(entry) {
		if (!filterTerm) return true;
		const term = filterTerm;
		const name = (entry.name || "").toLowerCase();
		const pidText = String(entry.pid || "");
		return name.includes(term) || pidText.includes(term);
	}

	function render(data) {
		const { system_prediction, items } = data;
		sysPredEl.textContent = fmt(system_prediction);
		tbody.innerHTML = "";
		for (const p of items.filter(matchesFilter)) {
			const tr = document.createElement("tr");
			tr.innerHTML = `
				<td>${p.pid}</td>
				<td>${p.name}</td>
				<td>${fmt(p.cpu_percent, 1)}</td>
				<td>${fmt(p.memory_percent, 1)}</td>
				<td>${fmt(p.cpu_time, 1)}</td>
				<td>${fmt(p.predicted_turnaround_time, 2)}</td>
				<td title="${p.turnaround_status ?? ""}">${fmt(p.actual_turnaround_time, 2)}</td>
				<td><button data-pid="${p.pid}">Kill</button></td>
			`;
			const btn = tr.querySelector("button");
			btn.addEventListener("click", () => kill(p.pid));
			tbody.appendChild(tr);
		}
	}

	function connect() {
		const proto = location.protocol === "https:" ? "wss:" : "ws:";
		const ws = new WebSocket(`${proto}//${location.host}/ws?top_n=50&quantum=3`);
		let lastMsgAt = Date.now();

		ws.onopen = () => {
			statusEl.textContent = "Live";
		};
		ws.onmessage = (ev) => {
			lastMsgAt = Date.now();
			try {
				const payload = JSON.parse(ev.data);
				lastPayload = payload;
				render(payload);
			} catch (e) {
				console.error("Bad WS payload", e);
			}
		};
		ws.onerror = () => {
			statusEl.textContent = "Error";
		};
		ws.onclose = () => {
			statusEl.textContent = "Reconnecting…";
			setTimeout(connect, 1200);
		};
	}

	if (searchInput) {
		searchInput.addEventListener("input", () => {
			filterTerm = searchInput.value.trim().toLowerCase();
			if (lastPayload) {
				render(lastPayload);
			}
		});
	}

	connect();
})();


