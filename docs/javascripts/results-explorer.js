(function () {
  "use strict";

  const root = document.getElementById("rmfs-results-explorer");
  if (!root) return;

  const state = {
    mode: "aggregate",
    dataset: "main",
    load: "high",
    metric: "completed_orders",
    reference: "Greedy",
    baseline: "all",
    search: "",
    sort: "arm",
    ascending: true,
  };

  const policyOrder = ["Greedy", "Hungarian", "JSQ", "WM-Base", "ComboS1J1"];
  let bundle = null;
  let currentRows = [];

  const esc = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const titleCase = (value) =>
    String(value).charAt(0).toUpperCase() + String(value).slice(1);

  function format(value, decimals) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function signed(value, decimals) {
    if (value === null || value === undefined) return "—";
    const number = Number(value);
    return `${number > 0 ? "+" : ""}${format(number, decimals)}`;
  }

  function availableMetrics(dataset, mode) {
    const keys = new Set();
    if (mode === "aggregate") {
      dataset.aggregate.forEach((row) => Object.keys(row.values).forEach((key) => keys.add(key)));
    } else {
      dataset.paired.forEach((row) => keys.add(row.metric));
    }
    return Object.keys(bundle.metrics).filter((key) => keys.has(key));
  }

  function metricOptions(metrics) {
    const groups = new Map();
    metrics.forEach((key) => {
      const item = bundle.metrics[key];
      if (!groups.has(item.group)) groups.set(item.group, []);
      groups.get(item.group).push([key, item]);
    });
    return Array.from(groups.entries())
      .map(
        ([group, items]) =>
          `<optgroup label="${esc(group)}">${items
            .map(
              ([key, item]) =>
                `<option value="${esc(key)}" ${state.metric === key ? "selected" : ""}>${esc(item.label)}</option>`
            )
            .join("")}</optgroup>`
      )
      .join("");
  }

  function ensureState() {
    const dataset = bundle.datasets[state.dataset];
    const metrics = availableMetrics(dataset, state.mode);
    if (!metrics.includes(state.metric)) {
      state.metric = metrics.includes("completed_orders") ? "completed_orders" : metrics[0];
    }
    if (state.mode === "aggregate") {
      const arms = dataset.aggregate
        .filter((row) => row.load === state.load && row.values[state.metric])
        .map((row) => row.arm);
      if (!arms.includes(state.reference)) {
        state.reference = arms.includes("Greedy") ? "Greedy" : arms[0];
      }
    }
  }

  function policyClass(arm, reference) {
    if (arm === "ComboS1J1") return "rmfs-series-proposed";
    if (arm === reference) return "rmfs-series-reference";
    const index = Math.max(0, policyOrder.indexOf(arm));
    return `rmfs-series-${index % 4}`;
  }

  function aggregateSvg(rows, metric, reference) {
    if (!rows.length) return '<div class="rmfs-empty">No rows match this selection.</div>';
    const width = 900;
    const left = 160;
    const right = 112;
    const top = 38;
    const rowHeight = 55;
    const bottom = 47;
    const height = top + rows.length * rowHeight + bottom;
    const extents = rows.flatMap((row) => {
      const value = row.values[metric];
      const spread = Number(value.std || 0);
      return [Number(value.mean) - spread, Number(value.mean) + spread];
    });
    let domainMin = Math.min(0, ...extents);
    let domainMax = Math.max(0, ...extents);
    const span = domainMax - domainMin || 1;
    domainMin -= span * 0.04;
    domainMax += span * 0.12;
    const chartWidth = width - left - right;
    const x = (value) => left + ((value - domainMin) / (domainMax - domainMin)) * chartWidth;
    const zeroX = x(0);
    const ticks = Array.from({ length: 6 }, (_, index) => domainMin + ((domainMax - domainMin) * index) / 5);
    const decimals = bundle.metrics[metric].decimals;
    const grid = ticks
      .map(
        (tick) => `<g class="rmfs-gridline">
          <line x1="${x(tick)}" y1="${top - 12}" x2="${x(tick)}" y2="${height - bottom + 4}" />
          <text x="${x(tick)}" y="${height - 14}" text-anchor="middle">${esc(format(tick, decimals > 1 ? 2 : 1))}</text>
        </g>`
      )
      .join("");
    const marks = rows
      .map((row, index) => {
        const value = row.values[metric];
        const mean = Number(value.mean);
        const spread = Number(value.std || 0);
        const y = top + index * rowHeight + rowHeight / 2;
        const xMean = x(mean);
        const xLow = x(mean - spread);
        const xHigh = x(mean + spread);
        const barX = Math.min(zeroX, xMean);
        const barWidth = Math.max(2, Math.abs(xMean - zeroX));
        const cssClass = policyClass(row.arm, reference);
        return `<g class="rmfs-chart-row ${cssClass}" tabindex="0">
          <title>${esc(row.arm)}: ${format(mean, decimals)} ± ${format(spread, decimals)} ${esc(bundle.metrics[metric].unit)}</title>
          <text class="rmfs-y-label" x="${left - 16}" y="${y + 5}" text-anchor="end">${esc(row.arm)}</text>
          <rect class="rmfs-bar" x="${barX}" y="${y - 11}" width="${barWidth}" height="22" rx="5" />
          <line class="rmfs-whisker" x1="${xLow}" y1="${y}" x2="${xHigh}" y2="${y}" />
          <line class="rmfs-whisker" x1="${xLow}" y1="${y - 6}" x2="${xLow}" y2="${y + 6}" />
          <line class="rmfs-whisker" x1="${xHigh}" y1="${y - 6}" x2="${xHigh}" y2="${y + 6}" />
          <circle class="rmfs-mean-dot" cx="${xMean}" cy="${y}" r="5" />
          <text class="rmfs-value-label" x="${Math.min(xMean + 10, width - right + 72)}" y="${y + 5}">${esc(format(mean, decimals))}</text>
        </g>`;
      })
      .join("");
    return `<svg class="rmfs-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Policy comparison chart">
      ${grid}
      <line class="rmfs-zero-line" x1="${zeroX}" y1="${top - 14}" x2="${zeroX}" y2="${height - bottom + 4}" />
      ${marks}
    </svg>`;
  }

  function favorable(row, bound) {
    return row.direction === "higher" ? Number(bound) > 0 : Number(bound) < 0;
  }

  function forestSvg(rows, metric) {
    if (!rows.length) return '<div class="rmfs-empty">No paired intervals match this selection.</div>';
    const width = 900;
    const left = 184;
    const right = 160;
    const top = 40;
    const rowHeight = 58;
    const bottom = 56;
    const height = top + rows.length * rowHeight + bottom;
    let domainMin = Math.min(0, ...rows.map((row) => Number(row.low)));
    let domainMax = Math.max(0, ...rows.map((row) => Number(row.high)));
    const span = domainMax - domainMin || 1;
    domainMin -= span * 0.12;
    domainMax += span * 0.12;
    const chartWidth = width - left - right;
    const x = (value) => left + ((value - domainMin) / (domainMax - domainMin)) * chartWidth;
    const zeroX = x(0);
    const decimals = bundle.metrics[metric].decimals;
    const ticks = Array.from({ length: 5 }, (_, index) => domainMin + ((domainMax - domainMin) * index) / 4);
    const grid = ticks
      .map(
        (tick) => `<g class="rmfs-gridline">
          <line x1="${x(tick)}" y1="${top - 13}" x2="${x(tick)}" y2="${height - bottom + 3}" />
          <text x="${x(tick)}" y="${height - 22}" text-anchor="middle">${esc(format(tick, decimals > 1 ? 2 : 1))}</text>
        </g>`
      )
      .join("");
    const marks = rows
      .map((row, index) => {
        const y = top + index * rowHeight + rowHeight / 2;
        const confirmed = row.direction === "higher" ? row.low > 0 : row.high < 0;
        const adverse = row.direction === "higher" ? row.high < 0 : row.low > 0;
        const cssClass = confirmed ? "rmfs-ci-confirmed" : adverse ? "rmfs-ci-adverse" : "rmfs-ci-uncertain";
        const label = `${titleCase(row.load)} · ${row.baseline}`;
        return `<g class="rmfs-ci-row ${cssClass}" tabindex="0">
          <title>${esc(label)}: ${signed(row.effect, decimals)} [${format(row.low, decimals)}, ${format(row.high, decimals)}]</title>
          <text class="rmfs-y-label" x="${left - 16}" y="${y + 5}" text-anchor="end">${esc(label)}</text>
          <line class="rmfs-ci-line" x1="${x(row.low)}" y1="${y}" x2="${x(row.high)}" y2="${y}" />
          <line class="rmfs-ci-cap" x1="${x(row.low)}" y1="${y - 7}" x2="${x(row.low)}" y2="${y + 7}" />
          <line class="rmfs-ci-cap" x1="${x(row.high)}" y1="${y - 7}" x2="${x(row.high)}" y2="${y + 7}" />
          <circle class="rmfs-ci-dot" cx="${x(row.effect)}" cy="${y}" r="6" />
          <text class="rmfs-value-label" x="${width - right + 16}" y="${y + 5}">${esc(signed(row.effect, decimals))}</text>
        </g>`;
      })
      .join("");
    return `<svg class="rmfs-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Paired bootstrap confidence interval plot">
      ${grid}
      <line class="rmfs-zero-line rmfs-zero-strong" x1="${zeroX}" y1="${top - 14}" x2="${zeroX}" y2="${height - bottom + 3}" />
      <text class="rmfs-zero-label" x="${zeroX}" y="17" text-anchor="middle">no difference</text>
      ${marks}
      <text class="rmfs-axis-title" x="${left + chartWidth / 2}" y="${height - 3}" text-anchor="middle">ComboS1J1 − baseline (${esc(bundle.metrics[metric].unit)})</text>
    </svg>`;
  }

  function controls(dataset, metrics) {
    const loads = ["low", "mid", "high"];
    const loadOptions = (state.mode === "paired" ? ["all", ...loads] : loads)
      .map(
        (load) => `<option value="${load}" ${state.load === load ? "selected" : ""}>${load === "all" ? "All loads" : titleCase(load)}</option>`
      )
      .join("");
    let fourthControl = "";
    if (state.mode === "aggregate") {
      const arms = policyOrder.filter((arm) => dataset.aggregate.some((row) => row.arm === arm));
      fourthControl = `<label>Reference policy
        <select data-field="reference">${arms
          .map((arm) => `<option value="${esc(arm)}" ${state.reference === arm ? "selected" : ""}>${esc(arm)}</option>`)
          .join("")}</select>
      </label>`;
    } else {
      const baselines = Array.from(new Set(dataset.paired.map((row) => row.baseline))).sort();
      fourthControl = `<label>Baseline
        <select data-field="baseline">
          <option value="all" ${state.baseline === "all" ? "selected" : ""}>All available</option>
          ${baselines
            .map((baseline) => `<option value="${esc(baseline)}" ${state.baseline === baseline ? "selected" : ""}>${esc(baseline)}</option>`)
            .join("")}
        </select>
      </label>`;
    }
    return `<div class="rmfs-controls" aria-label="Result filters">
      <label>Evaluation
        <select data-field="dataset">
          ${Object.entries(bundle.datasets)
            .map(([key, item]) => `<option value="${key}" ${state.dataset === key ? "selected" : ""}>${esc(item.label)}</option>`)
            .join("")}
        </select>
      </label>
      <label>Load
        <select data-field="load">${loadOptions}</select>
      </label>
      <label>Metric
        <select data-field="metric">${metricOptions(metrics)}</select>
      </label>
      ${fourthControl}
    </div>`;
  }

  function summaryCardsAggregate(rows, metric, reference) {
    const meta = bundle.metrics[metric];
    const direction = meta.direction;
    const best = rows.reduce((winner, row) => {
      if (!winner) return row;
      const value = row.values[metric].mean;
      const current = winner.values[metric].mean;
      return direction === "higher" ? (value > current ? row : winner) : value < current ? row : winner;
    }, null);
    const proposed = rows.find((row) => row.arm === "ComboS1J1");
    const referenceRow = rows.find((row) => row.arm === reference);
    const delta = proposed && referenceRow ? proposed.values[metric].mean - referenceRow.values[metric].mean : null;
    return `<div class="rmfs-summary-grid">
      <div class="rmfs-summary-card"><span>Best descriptive mean</span><strong>${esc(best?.arm || "—")}</strong><small>${best ? format(best.values[metric].mean, meta.decimals) : "—"} ${esc(meta.unit)}</small></div>
      <div class="rmfs-summary-card rmfs-accent-card"><span>ComboS1J1 mean</span><strong>${proposed ? format(proposed.values[metric].mean, meta.decimals) : "—"}</strong><small>${esc(meta.unit)}</small></div>
      <div class="rmfs-summary-card"><span>Δ vs ${esc(reference)}</span><strong>${signed(delta, meta.decimals)}</strong><small>raw mean difference</small></div>
      <div class="rmfs-summary-card"><span>Interpretation</span><strong>${titleCase(direction)} is better</strong><small>mean ± SD across seeds</small></div>
    </div>`;
  }

  function summaryCardsPaired(rows, metric) {
    const meta = bundle.metrics[metric];
    const confirmed = rows.filter((row) => (row.direction === "higher" ? row.low > 0 : row.high < 0)).length;
    const uncertain = rows.filter((row) => row.low <= 0 && row.high >= 0).length;
    const seeds = Array.from(new Set(rows.map((row) => row.n))).sort((a, b) => a - b);
    return `<div class="rmfs-summary-grid">
      <div class="rmfs-summary-card rmfs-accent-card"><span>Displayed intervals</span><strong>${rows.length}</strong><small>paired bootstrap 95% CIs</small></div>
      <div class="rmfs-summary-card"><span>Favorable, excludes zero</span><strong>${confirmed}</strong><small>using the declared direction</small></div>
      <div class="rmfs-summary-card"><span>Intervals crossing zero</span><strong>${uncertain}</strong><small>not confirmed pairwise effects</small></div>
      <div class="rmfs-summary-card"><span>Paired seeds</span><strong>${seeds.length ? seeds.join(" / ") : "—"}</strong><small>${esc(meta.label)}</small></div>
    </div>`;
  }

  function sortRows(rows, mode) {
    const factor = state.ascending ? 1 : -1;
    return [...rows].sort((a, b) => {
      let left;
      let right;
      if (mode === "aggregate") {
        left = state.sort === "arm" ? a.arm : a.values[state.metric]?.[state.sort];
        right = state.sort === "arm" ? b.arm : b.values[state.metric]?.[state.sort];
      } else {
        left = state.sort === "label" ? `${a.load}-${a.baseline}` : a[state.sort];
        right = state.sort === "label" ? `${b.load}-${b.baseline}` : b[state.sort];
      }
      if (typeof left === "number" && typeof right === "number") return factor * (left - right);
      return factor * String(left ?? "").localeCompare(String(right ?? ""));
    });
  }

  function sortButton(label, key) {
    const active = state.sort === key;
    const arrow = active ? (state.ascending ? " ↑" : " ↓") : "";
    return `<button type="button" class="rmfs-sort ${active ? "active" : ""}" data-sort="${key}">${esc(label + arrow)}</button>`;
  }

  function aggregateTable(rows, metric, reference) {
    const meta = bundle.metrics[metric];
    const referenceRow = rows.find((row) => row.arm === reference);
    const filtered = rows.filter((row) => row.arm.toLowerCase().includes(state.search.toLowerCase()));
    const sorted = sortRows(filtered, "aggregate");
    currentRows = sorted.map((row) => ({
      load: row.load,
      policy: row.arm,
      metric,
      mean: row.values[metric].mean,
      std: row.values[metric].std,
      n: row.values[metric].n,
      reference,
      delta_vs_reference: referenceRow
        ? row.values[metric].mean - referenceRow.values[metric].mean
        : null,
    }));
    return `<div class="rmfs-table-wrap"><table class="rmfs-data-table">
      <thead><tr>
        <th>${sortButton("Policy", "arm")}</th>
        <th>${sortButton("Mean", "mean")}</th>
        <th>${sortButton("SD", "std")}</th>
        <th>${sortButton("n", "n")}</th>
        <th>Δ vs ${esc(reference)}</th>
      </tr></thead>
      <tbody>${sorted
        .map((row) => {
          const value = row.values[metric];
          const delta = referenceRow ? value.mean - referenceRow.values[metric].mean : null;
          const favorableDelta = meta.direction === "higher" ? delta > 0 : delta < 0;
          return `<tr class="${row.arm === "ComboS1J1" ? "rmfs-proposed-row" : ""}">
            <td><span class="rmfs-policy-dot ${policyClass(row.arm, reference)}"></span>${esc(row.arm)}</td>
            <td>${format(value.mean, meta.decimals)}</td>
            <td>${format(value.std, meta.decimals)}</td>
            <td>${format(value.n, 0)}</td>
            <td class="${delta === 0 ? "" : favorableDelta ? "rmfs-good" : "rmfs-caution"}">${signed(delta, meta.decimals)}</td>
          </tr>`;
        })
        .join("")}</tbody>
    </table></div>`;
  }

  function pairedTable(rows, metric) {
    const meta = bundle.metrics[metric];
    const filtered = rows.filter((row) =>
      `${row.load} ${row.baseline}`.toLowerCase().includes(state.search.toLowerCase())
    );
    const sorted = sortRows(filtered, "paired");
    currentRows = sorted.map((row) => ({
      load: row.load,
      proposed: row.proposed,
      baseline: row.baseline,
      metric: row.metric,
      direction: row.direction,
      effect: row.effect,
      ci95_low: row.low,
      ci95_high: row.high,
      wins: row.wins,
      losses: row.losses,
      ties: row.ties,
      paired_seeds: row.n,
    }));
    return `<div class="rmfs-table-wrap"><table class="rmfs-data-table">
      <thead><tr>
        <th>${sortButton("Load / baseline", "label")}</th>
        <th>${sortButton("Effect", "effect")}</th>
        <th>Paired 95% CI</th>
        <th>W / L / T</th>
        <th>${sortButton("n", "n")}</th>
      </tr></thead>
      <tbody>${sorted
        .map((row) => {
          const confirmed = row.direction === "higher" ? row.low > 0 : row.high < 0;
          return `<tr>
            <td><strong>${esc(titleCase(row.load))}</strong><br><small>vs ${esc(row.baseline)}</small></td>
            <td class="${confirmed ? "rmfs-good" : ""}">${signed(row.effect, meta.decimals)}</td>
            <td>[${format(row.low, meta.decimals)}, ${format(row.high, meta.decimals)}]</td>
            <td>${row.wins} / ${row.losses} / ${row.ties}</td>
            <td>${row.n}</td>
          </tr>`;
        })
        .join("")}</tbody>
    </table></div>`;
  }

  function render() {
    ensureState();
    const dataset = bundle.datasets[state.dataset];
    const metrics = availableMetrics(dataset, state.mode);
    const meta = bundle.metrics[state.metric];
    let rows;
    let chart;
    let cards;
    let table;
    let uncertainty;
    if (state.mode === "aggregate") {
      if (state.load === "all") state.load = "high";
      rows = dataset.aggregate.filter(
        (row) => row.load === state.load && row.values[state.metric]
      );
      const searched = rows.filter((row) => row.arm.toLowerCase().includes(state.search.toLowerCase()));
      chart = aggregateSvg(searched, state.metric, state.reference);
      cards = summaryCardsAggregate(rows, state.metric, state.reference);
      table = aggregateTable(rows, state.metric, state.reference);
      uncertainty = "Bars show means; whiskers show ±1 SD across seeds.";
    } else {
      rows = dataset.paired.filter(
        (row) =>
          row.metric === state.metric &&
          (state.load === "all" || row.load === state.load) &&
          (state.baseline === "all" || row.baseline === state.baseline)
      );
      const searched = rows.filter((row) =>
        `${row.load} ${row.baseline}`.toLowerCase().includes(state.search.toLowerCase())
      );
      chart = forestSvg(searched, state.metric);
      cards = summaryCardsPaired(rows, state.metric);
      table = pairedTable(rows, state.metric);
      uncertainty = "Points are paired mean effects; lines are bootstrap 95% CIs.";
    }

    root.innerHTML = `<div class="rmfs-explorer-head">
      <div>
        <span class="rmfs-eyebrow">FROZEN EVIDENCE · INTERACTIVE VIEW</span>
        <h2>Policy outcomes, without the spreadsheet work</h2>
        <p>${esc(dataset.detail)}</p>
      </div>
      <div class="rmfs-mode-switch" role="group" aria-label="Analysis view">
        <button type="button" data-mode="aggregate" class="${state.mode === "aggregate" ? "active" : ""}">Aggregate means</button>
        <button type="button" data-mode="paired" class="${state.mode === "paired" ? "active" : ""}">Paired effects</button>
      </div>
    </div>
    ${controls(dataset, metrics)}
    ${cards}
    <section class="rmfs-visual-panel" aria-labelledby="rmfs-chart-title">
      <div class="rmfs-panel-heading">
        <div><span class="rmfs-kicker">${state.mode === "aggregate" ? "DESCRIPTIVE COMPARISON" : "PAIRED INFERENCE"}</span>
        <h3 id="rmfs-chart-title">${esc(meta.label)} · ${state.load === "all" ? "all loads" : titleCase(state.load)}</h3></div>
        <span class="rmfs-direction">${titleCase(meta.direction)} is better</span>
      </div>
      <div class="rmfs-chart-frame">${chart}</div>
      <p class="rmfs-chart-note">${esc(uncertainty)}</p>
    </section>
    <section class="rmfs-visual-panel rmfs-table-panel" aria-labelledby="rmfs-table-title">
      <div class="rmfs-panel-heading rmfs-table-heading">
        <div><span class="rmfs-kicker">AUDIT THE VALUES</span><h3 id="rmfs-table-title">Filtered data</h3></div>
        <div class="rmfs-table-actions">
          <label class="rmfs-search"><span class="sr-only">Search rows</span><input type="search" data-field="search" value="${esc(state.search)}" placeholder="Filter policies…"></label>
          <button type="button" class="rmfs-download-button" data-download>Download current view</button>
        </div>
      </div>
      ${table}
    </section>`;
  }

  function downloadCurrentView() {
    if (!currentRows.length) return;
    const fields = Object.keys(currentRows[0]);
    const quote = (value) => {
      if (value === null || value === undefined) return "";
      const text = String(value);
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    const csv = [fields.join(","), ...currentRows.map((row) => fields.map((field) => quote(row[field])).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `rmfs-${state.dataset}-${state.mode}-${state.metric}.csv`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }

  root.addEventListener("change", (event) => {
    const field = event.target.dataset.field;
    if (!field) return;
    state[field] = event.target.value;
    if (["dataset", "metric"].includes(field)) {
      state.sort = state.mode === "aggregate" ? "arm" : "label";
    }
    render();
  });

  root.addEventListener("input", (event) => {
    if (event.target.dataset.field !== "search") return;
    state.search = event.target.value;
    render();
    const input = root.querySelector('[data-field="search"]');
    if (input) {
      input.focus();
      input.setSelectionRange(state.search.length, state.search.length);
    }
  });

  root.addEventListener("click", (event) => {
    const modeButton = event.target.closest("[data-mode]");
    if (modeButton) {
      state.mode = modeButton.dataset.mode;
      state.load = "high";
      state.metric = "completed_orders";
      state.sort = state.mode === "aggregate" ? "arm" : "label";
      state.search = "";
      render();
      return;
    }
    const sortButton = event.target.closest("[data-sort]");
    if (sortButton) {
      const key = sortButton.dataset.sort;
      if (state.sort === key) state.ascending = !state.ascending;
      else {
        state.sort = key;
        state.ascending = true;
      }
      render();
      return;
    }
    if (event.target.closest("[data-download]")) downloadCurrentView();
  });

  const source = new URL(root.dataset.source, document.baseURI);
  fetch(source)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      bundle = data;
      render();
    })
    .catch((error) => {
      root.innerHTML = `<div class="rmfs-error" role="alert"><strong>Could not load the result bundle.</strong><br>${esc(error.message)}. Use the direct CSV downloads below.</div>`;
    });
})();
