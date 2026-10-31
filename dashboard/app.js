/* Mumbai AQI Forecaster dashboard. Reads the CSVs the pipeline writes to dashboard/data/. */
(() => {
  const state = { station: 'City average', data: null };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const parse = d3.timeParse('%Y-%m-%dT%H:%M');
  const fmtTick = d3.timeFormat('%a %-I %p');
  const fmtTip = d3.timeFormat('%a %b %-d, %-I %p');
  const tip = document.getElementById('tip');

  const catOf = (a) => a <= 50 ? 'Good' : a <= 100 ? 'Satisfactory' : a <= 200 ? 'Moderate' : a <= 300 ? 'Poor' : a <= 400 ? 'Very poor' : 'Severe';
  const catClass = (c) => 'cat-' + String(c).toLowerCase().replace(/\s+/g, '-');
  const catColor = (c) => ({ Good: css('--ok'), Satisfactory: css('--ok'), Moderate: css('--warn') })[c] || css('--bad');
  const setText = (id, v) => { document.getElementById(id).textContent = v; };

  function showTip(lines, x, y) {
    tip.replaceChildren(...lines.map(([text, muted]) => {
      const d = document.createElement('div');
      if (muted) d.className = 'k';
      d.textContent = text;
      return d;
    }));
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14; if (left + w > window.innerWidth - 8) left = x - w - 14;
    let top = y - h - 10; if (top < 8) top = y + 14;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = top + 'px';
  }
  const hideTip = () => { tip.hidden = true; };

  function measure(svg, texts, cls = 'tick-label') {
    const t = svg.append('text').attr('class', cls).attr('opacity', 0);
    let w = 0;
    for (const s of texts) { t.text(s); w = Math.max(w, t.node().getComputedTextLength()); }
    t.remove();
    return w;
  }

  // ---------- derived numbers ----------
  function summaries(hourly) {
    const out = {};
    for (const [station, rs] of d3.group(hourly, (r) => r.station)) {
      const sorted = [...rs].sort((a, b) => a.t - b.t);
      const act = sorted.filter((r) => r.kind === 'actual');
      const fc = sorted.filter((r) => r.kind === 'forecast').slice(0, 24);
      const cur = act[act.length - 1];
      const pk = d3.greatest(fc, (r) => r.aqi);
      out[station] = { current: cur?.aqi, peak: pk?.aqi, peakTime: pk ? fmtTick(pk.t) : '—' };
    }
    return out;
  }

  // ---------- header and tiles ----------
  function renderHeader() {
    const { summary, stations, metrics, meta } = state.data;
    const s = summary[state.station];
    if (!s) return;
    const take = document.getElementById('takeaway');
    take.replaceChildren();
    const b = (t) => { const e = document.createElement('b'); e.textContent = t; return e; };
    take.append(state.station + ' is at AQI ', b(s.current), ' right now. The model expects it to peak at ', b(s.peak), ' around ', b(s.peakTime), '.');

    setText('t-current', s.current);
    const cc = document.getElementById('t-current-cat'); cc.textContent = catOf(s.current); cc.className = 'chip ' + catClass(catOf(s.current));
    setText('t-peak', s.peak);
    const pc = document.getElementById('t-peak-cat'); pc.textContent = catOf(s.peak); pc.className = 'chip ' + catClass(catOf(s.peak));
    setText('t-peak-time', s.peakTime);

    const poor = stations.filter((r) => r.aqi > 200).length;
    const worst = d3.greatest(stations, (r) => r.aqi);
    setText('t-poor', poor);
    setText('t-worst', worst ? `${worst.station} at ${worst.aqi}` : '—');

    const m = (model, h) => metrics.find((r) => r.model === model && r.horizon === h);
    const lgb = m('LightGBM + weather', '24h'), base = m('Persistence baseline', '24h');
    setText('t-mae', lgb ? lgb.mae : '—');
    if (lgb && base) {
      const pct = Math.round(100 * (1 - lgb.mae / base.mae));
      setText('t-lift', pct >= 0 ? `${pct}% lower than the persistence baseline` : `${-pct}% higher than the persistence baseline`);
    }

    const note = document.getElementById('source-note');
    note.replaceChildren();
    const badge = document.createElement('span');
    const live = meta.source === 'live';
    badge.className = 'badge ' + (live ? 'live' : 'sample');
    badge.textContent = live ? 'Live data' : 'Sample data';
    note.append(badge, live
      ? `Latest reading ${meta.latest_reading || ''}, updated ${new Date(meta.generated_at).toLocaleString()}`
      : 'Simulated readings for demonstration. Run the pipeline with an OpenAQ key for real data.');
  }

  // ---------- forecast chart ----------
  function renderForecast() {
    const el = document.getElementById('chart-forecast');
    const rows = state.data.hourly.filter((r) => r.station === state.station).sort((a, b) => a.t - b.t);
    el.replaceChildren();
    if (!rows.length) return;
    const act = rows.filter((r) => r.kind === 'actual'), fc = rows.filter((r) => r.kind === 'forecast');
    const now = act[act.length - 1];
    const W = el.clientWidth, H = 320, bottom = 28;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const y = d3.scaleLinear([0, d3.max(rows, (r) => Math.max(r.aqi, r.hi ?? 0)) * 1.12], [H - bottom, 12]).nice();
    const yt = y.ticks(5);
    const m = { l: measure(svg, yt.map(String)) + 10, r: 76 };
    const x = d3.scaleTime(d3.extent(rows, (r) => r.t), [m.l, W - m.r]);
    const accent = css('--accent');

    svg.append('g').selectAll('line').data(yt).join('line')
      .attr('x1', m.l).attr('x2', W - m.r).attr('y1', y).attr('y2', y).attr('stroke', css('--grid'));
    svg.append('g').selectAll('text').data(yt).join('text').attr('class', 'tick-label')
      .attr('x', m.l - 8).attr('y', y).attr('dy', '0.32em').attr('text-anchor', 'end').text(String);
    for (const [v, name] of [[100, 'Moderate'], [200, 'Poor'], [300, 'Very poor']]) {
      if (v >= y.domain()[1]) continue;
      svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', css('--muted')).attr('stroke-opacity', 0.5).attr('stroke-dasharray', '4 4');
      svg.append('text').attr('class', 'tick-label').attr('x', W - m.r + 6).attr('y', y(v)).attr('dy', '0.32em').text(`${name} ${v}`);
    }
    const xt = x.ticks(Math.max(2, Math.min(7, Math.floor((W - m.l - m.r) / 110))));
    svg.append('g').selectAll('text').data(xt).join('text').attr('class', 'tick-label')
      .attr('x', x).attr('y', H - 8).attr('text-anchor', 'middle').text(fmtTick);

    const band = [{ t: now.t, lo: now.aqi, hi: now.aqi }, ...fc];
    svg.append('path').datum(band).attr('fill', accent).attr('fill-opacity', 0.18)
      .attr('d', d3.area().x((d) => x(d.t)).y0((d) => y(d.lo)).y1((d) => y(d.hi)));
    const line = d3.line().x((d) => x(d.t)).y((d) => y(d.aqi));
    svg.append('path').datum(act).attr('fill', 'none').attr('stroke', accent).attr('stroke-width', 2)
      .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('d', line);
    svg.append('path').datum([now, ...fc]).attr('fill', 'none').attr('stroke', accent).attr('stroke-width', 2)
      .attr('stroke-dasharray', '5 4').attr('d', line);
    svg.append('line').attr('x1', x(now.t)).attr('x2', x(now.t)).attr('y1', 12).attr('y2', H - bottom).attr('stroke', css('--muted'));
    svg.append('text').attr('class', 'tick-label').attr('x', x(now.t) + 6).attr('y', 22).text('Now');

    const dot = svg.append('circle').attr('r', 4).attr('fill', accent).attr('stroke', css('--panel')).attr('stroke-width', 2).style('display', 'none');
    const rule = svg.append('line').attr('y1', 12).attr('y2', H - bottom).attr('stroke', css('--muted')).attr('stroke-dasharray', '2 3').style('display', 'none');
    const bis = d3.bisector((d) => d.t).center;
    svg.append('rect').attr('x', m.l).attr('width', Math.max(0, W - m.l - m.r)).attr('height', H - bottom).attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const d = rows[bis(rows, x.invert(d3.pointer(ev)[0]))];
        dot.style('display', null).attr('cx', x(d.t)).attr('cy', y(d.aqi));
        rule.style('display', null).attr('x1', x(d.t)).attr('x2', x(d.t));
        const lines = [[fmtTip(d.t), true], [`${d.kind === 'actual' ? 'Measured' : 'Forecast'} ${d.aqi} · ${catOf(d.aqi)}`]];
        if (d.lo != null) lines.push([`Range ${d.lo}–${d.hi}`, true]);
        showTip(lines, ev.clientX, ev.clientY);
      })
      .on('pointerleave', () => { dot.style('display', 'none'); rule.style('display', 'none'); hideTip(); });
  }

  // ---------- weather chart ----------
  function renderWeather() {
    const el = document.getElementById('chart-weather');
    const rows = state.data.hourly.filter((r) => r.station === state.station).sort((a, b) => a.t - b.t);
    el.replaceChildren();
    if (!rows.length) return;
    const W = el.clientWidth, H = 230, panelH = 82;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const series = [
      { key: 'wind', name: 'Wind speed (km/h)', color: css('--accent-2'), top: 18 },
      { key: 'hum', name: 'Relative humidity (%)', color: css('--accent-3'), top: 18 + panelH + 30 },
    ];
    const allTicks = series.flatMap((s) => d3.scaleLinear(d3.extent(rows, (r) => r[s.key])).nice().ticks(3).map(String));
    const m = { l: measure(svg, allTicks) + 10, r: 16 };
    const x = d3.scaleTime(d3.extent(rows, (r) => r.t), [m.l, W - m.r]);
    const nowT = rows.filter((r) => r.kind === 'actual').pop().t;
    const dots = [];
    for (const s of series) {
      const y = d3.scaleLinear(d3.extent(rows, (r) => r[s.key]), [s.top + panelH, s.top]).nice();
      const g = svg.append('g');
      g.append('text').attr('x', m.l).attr('y', s.top - 6).attr('font-size', 12).attr('fill', css('--fg')).text(s.name);
      const yt = y.ticks(3);
      g.selectAll('line.g').data(yt).join('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y).attr('y2', y).attr('stroke', css('--grid'));
      g.selectAll('text.t').data(yt).join('text').attr('class', 'tick-label').attr('x', m.l - 8).attr('y', y).attr('dy', '0.32em').attr('text-anchor', 'end').text(String);
      g.append('path').datum(rows).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 2).attr('stroke-linejoin', 'round')
        .attr('d', d3.line().x((d) => x(d.t)).y((d) => y(d[s.key])));
      g.append('line').attr('x1', x(nowT)).attr('x2', x(nowT)).attr('y1', s.top).attr('y2', s.top + panelH).attr('stroke', css('--muted'));
      dots.push({ s, y, c: g.append('circle').attr('r', 3.5).attr('fill', s.color).style('display', 'none') });
    }
    const xt = x.ticks(Math.max(2, Math.min(7, Math.floor((W - m.l - m.r) / 110))));
    svg.append('g').selectAll('text').data(xt).join('text').attr('class', 'tick-label').attr('x', x).attr('y', H - 6).attr('text-anchor', 'middle').text(fmtTick);
    const bis = d3.bisector((d) => d.t).center;
    svg.append('rect').attr('x', m.l).attr('width', Math.max(0, W - m.l - m.r)).attr('height', H - 20).attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const d = rows[bis(rows, x.invert(d3.pointer(ev)[0]))];
        for (const o of dots) o.c.style('display', null).attr('cx', x(d.t)).attr('cy', o.y(d[o.s.key]));
        showTip([[fmtTip(d.t), true], [`Wind ${d.wind} km/h`], [`Humidity ${d.hum}%`]], ev.clientX, ev.clientY);
      })
      .on('pointerleave', () => { for (const o of dots) o.c.style('display', 'none'); hideTip(); });
  }

  // ---------- station map ----------
  function renderMap() {
    const el = document.getElementById('chart-map');
    const rows = state.data.stations;
    el.replaceChildren();
    if (!rows.length) return;
    const W = el.clientWidth;
    // Keep the region's real proportions: 1° of longitude here is ~0.945 × 1° of latitude
    const lonSpan = (d3.max(rows, (d) => d.lon) - d3.min(rows, (d) => d.lon)) * 0.945;
    const latSpan = d3.max(rows, (d) => d.lat) - d3.min(rows, (d) => d.lat);
    const pad = 70, top = 26, bottom = 34;
    const H = Math.round(Math.min(560, Math.max(360, (W - 2 * pad) * (latSpan / Math.max(lonSpan, 1e-6)) + top + bottom)));
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const scale = Math.min((W - 2 * pad) / Math.max(lonSpan, 1e-6), (H - top - bottom) / Math.max(latSpan, 1e-6));
    const cx = (d3.min(rows, (d) => d.lon) + d3.max(rows, (d) => d.lon)) / 2;
    const cy = (d3.min(rows, (d) => d.lat) + d3.max(rows, (d) => d.lat)) / 2;
    const x = (lon) => W / 2 + (lon - cx) * 0.945 * scale;
    const y = (lat) => top + (H - top - bottom) / 2 - (lat - cy) * scale;
    const r = d3.scaleSqrt([0, d3.max(rows, (d) => d.aqi)], [0, rows.length > 15 ? 10 : 13]);

    svg.append('text').attr('class', 'tick-label').attr('x', 2).attr('y', 14).text('N ↑');
    svg.append('text').attr('class', 'tick-label').attr('x', 2).attr('y', H - 6).text('← Arabian Sea      Thane Creek →');

    // Place labels greedily: selected station first, then worst AQI; skip any that would overlap
    const placed = rows.map((d) => ({ x0: x(d.lon) - r(d.aqi), x1: x(d.lon) + r(d.aqi), y0: y(d.lat) - r(d.aqi), y1: y(d.lat) + r(d.aqi) }));
    const probe = svg.append('text').attr('font-size', 12).attr('opacity', 0);
    const labels = new Map();
    const order = [...rows].sort((a, b) => (b.station === state.station) - (a.station === state.station) || b.aqi - a.aqi);
    for (const d of order) {
      probe.text(d.area);
      const w = probe.node().getComputedTextLength(), px = x(d.lon), py = y(d.lat), gap = r(d.aqi) + 4;
      for (const side of [1, -1]) {
        const box = side === 1 ? { x0: px + gap, x1: px + gap + w } : { x0: px - gap - w, x1: px - gap };
        Object.assign(box, { y0: py - 8, y1: py + 8 });
        if (box.x0 < 0 || box.x1 > W) continue;
        // A circle's own box is in `placed`, so ignore overlap with itself
        const own = placed[rows.indexOf(d)];
        const clash = placed.some((p) => p !== own && box.x0 < p.x1 && box.x1 > p.x0 && box.y0 < p.y1 && box.y1 > p.y0);
        if (!clash) { labels.set(d.station, side); placed.push(box); break; }
      }
    }
    probe.remove();

    const g = svg.selectAll('g.pt').data(rows).join('g').attr('class', 'pt').attr('tabindex', 0).attr('role', 'button')
      .attr('aria-label', (d) => `${d.station}, AQI ${d.aqi}`)
      .attr('transform', (d) => `translate(${x(d.lon)},${y(d.lat)})`);
    g.append('circle').attr('r', (d) => r(d.aqi)).attr('fill', (d) => catColor(d.category)).attr('fill-opacity', 0.25)
      .attr('stroke', (d) => catColor(d.category)).attr('stroke-width', (d) => (d.station === state.station ? 3 : 1.5));
    g.append('circle').attr('r', 2.5).attr('fill', (d) => catColor(d.category));
    g.filter((d) => labels.has(d.station)).append('text')
      .attr('x', (d) => labels.get(d.station) * (r(d.aqi) + 4)).attr('text-anchor', (d) => (labels.get(d.station) === 1 ? 'start' : 'end'))
      .attr('dy', '0.32em').attr('font-size', 12).attr('font-weight', (d) => (d.station === state.station ? 600 : 400))
      .attr('fill', css('--fg')).text((d) => d.area);
    g.on('click', (ev, d) => select(d.station === state.station ? 'City average' : d.station))
      .on('keydown', (ev, d) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(d.station); } })
      .on('pointerenter', (ev, d) => showTip([[d.station, true], [`AQI ${d.aqi} · ${d.category}`], [`PM2.5 ${d.pm25 ?? '—'} · PM10 ${d.pm10 ?? '—'} µg/m³ (24h)`, true]], ev.clientX, ev.clientY))
      .on('pointerleave', hideTip);
  }

  // ---------- SHAP drivers ----------
  function renderDrivers() {
    const el = document.getElementById('chart-drivers');
    const rows = [...state.data.drivers].sort((a, b) => b.v - a.v);
    el.replaceChildren();
    if (!rows.length) return;
    const W = el.clientWidth, rowH = 30, H = rows.length * rowH + 8;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const lw = Math.min(measure(svg, rows.map((d) => d.f), '') + 12, W * 0.5);
    const vw = measure(svg, rows.map((d) => d.v.toFixed(1))) + 8;
    const x = d3.scaleLinear([0, d3.max(rows, (d) => d.v)], [lw, W - vw]);
    const y = d3.scaleBand(rows.map((d) => d.f), [4, H - 4]).padding(0.3);
    const g = svg.selectAll('g').data(rows).join('g');
    g.append('text').attr('x', lw - 10).attr('y', (d) => y(d.f) + y.bandwidth() / 2).attr('dy', '0.32em').attr('text-anchor', 'end')
      .attr('font-size', 12).attr('fill', css('--fg')).text((d) => d.f);
    g.append('rect').attr('x', lw).attr('y', (d) => y(d.f)).attr('height', y.bandwidth()).attr('rx', 3)
      .attr('width', (d) => Math.max(1, x(d.v) - lw)).attr('fill', (d, i) => (i < 2 ? css('--accent') : css('--line')));
    g.append('text').attr('class', 'tick-label').attr('x', (d) => x(d.v) + 6).attr('y', (d) => y(d.f) + y.bandwidth() / 2).attr('dy', '0.32em').text((d) => d.v.toFixed(1));
  }

  // ---------- tables ----------
  function cellRow(values, cls = '') {
    const tr = document.createElement('tr');
    tr.className = cls;
    for (const [v, c] of values) { const td = document.createElement('td'); if (c) td.className = c; td.textContent = v; tr.appendChild(td); }
    return tr;
  }

  function renderAlerts() {
    const tb = document.getElementById('alerts-body');
    tb.replaceChildren(...[...state.data.stations].sort((a, b) => b.aqi - a.aqi).map((r) => {
      const tr = cellRow([[r.station], [r.area, 'hide-narrow'], [r.aqi, 'num'], [r.category, catClass(r.category)], [r.dominant, 'hide-narrow'], [r.advice]],
        'click' + (r.station === state.station ? ' sel' : ''));
      tr.addEventListener('click', () => select(r.station === state.station ? 'City average' : r.station));
      return tr;
    }));
  }

  function renderMetrics() {
    const rows = state.data.metrics;
    const best = {};
    for (const r of rows) if (!best[r.horizon] || r.mae < best[r.horizon].mae) best[r.horizon] = r;
    const order = ['24h', '48h', '72h'];
    const sorted = [...rows].sort((a, b) => order.indexOf(a.horizon) - order.indexOf(b.horizon) || b.mae - a.mae);
    document.getElementById('metrics-body').replaceChildren(...sorted.map((r) =>
      cellRow([[r.model], [r.horizon], [r.mae.toFixed(1), 'num'], [r.rmse.toFixed(1), 'num']], best[r.horizon] === r ? 'best' : '')));
  }

  // ---------- wiring ----------
  function renderAll() {
    renderHeader(); renderForecast(); renderWeather(); renderMap(); renderDrivers(); renderAlerts(); renderMetrics();
  }

  function select(station) {
    state.station = station;
    document.getElementById('station').value = station;
    history.replaceState(null, '', station === 'City average' ? location.pathname : '#' + encodeURIComponent(station));
    renderAll();
  }

  async function load() {
    const num = (v) => (v === '' || v == null ? null : +v);
    const [hourly, stations, drivers, metrics, meta] = await Promise.all([
      d3.csv('data/aqi_hourly.csv', (r) => ({ station: r.station, t: parse(r.ts), kind: r.kind, aqi: num(r.aqi), lo: num(r.aqi_low), hi: num(r.aqi_high), wind: num(r.wind_kmh), hum: num(r.humidity_pct) })),
      d3.csv('data/aqi_stations.csv', (r) => ({ ...r, lat: +r.lat, lon: +r.lon, aqi: +r.aqi, pm25: num(r.pm25), pm10: num(r.pm10) })),
      d3.csv('data/aqi_drivers.csv', (r) => ({ f: r.feature, v: +r.mean_abs_shap })),
      d3.csv('data/aqi_model_metrics.csv', (r) => ({ model: r.model, horizon: r.horizon, mae: +r.mae, rmse: +r.rmse })),
      d3.json('data/meta.json').catch(() => ({ source: 'sample' })),
    ]);
    state.data = { hourly, stations, drivers, metrics, meta, summary: summaries(hourly) };

    const sel = document.getElementById('station');
    sel.replaceChildren(new Option('City average', 'City average'), ...[...stations].sort((a, b) => a.station.localeCompare(b.station)).map((s) => new Option(s.station, s.station)));
    sel.addEventListener('change', () => select(sel.value));
    const fromHash = decodeURIComponent(location.hash.slice(1));
    select(state.data.summary[fromHash] ? fromHash : 'City average');

    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(renderAll, 150); });
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', renderAll);
  }

  load().catch((err) => {
    const take = document.getElementById('takeaway');
    take.className = 'take error';
    take.textContent = 'Could not load data. Serve this folder over HTTP (python -m http.server) rather than opening the file directly. ' + err.message;
  });
})();