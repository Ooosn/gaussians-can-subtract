/* Gaussians Can Subtract - interactive figures (no dependencies) */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DATA = window.GAUSSOLVE_DATA || {};
  // release tag from index.html (data.js?v=N), so updated viewer images are never served stale
  const ASSET_V = ((document.querySelector('script[src*="data.js"]') || {}).src || '').split('v=')[1] || '';
  const h = (tag, attrs = {}, html = '') => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v == null) continue;
      el.setAttribute(k, v === true ? '' : v);
    }
    if (html) el.innerHTML = html;
    return el;
  };

  const ICON_ARROWS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l-6 6 6 6M15 6l6 6-6 6"/></svg>';
  const ICON_LENS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></svg>';
  const ICON_SWAP = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 7h12l-3-3M17 17H5l3 3"/></svg>';

  /* ------------------------------------------------------------------ sets */
  const SETS = {
    decomp: {
      base: 'assets/decomp/',
      scenes: DATA.decomp || [],
      layers: [
        { id: 'gt', label: 'Ground truth' },
        { id: 'mesh', label: 'Textured mesh' },
        { id: 'ndf', label: 'Dissolve field' },
        { id: 'residual', label: 'Residual Gaussians' },
        { id: 'ours', label: 'Gaussolve' },
      ],
      left: 'mesh', right: 'ours',
      steps: [
        { title: 'Mesh', left: 'gt', right: 'mesh',
          text: 'The textured mesh covers broad, regular surfaces, but drifts from the images where its geometry, occlusion or appearance is off.' },
        { title: 'Subtract', left: 'mesh', right: 'ndf',
          text: 'The Neural Dissolve Field marks the mesh content to remove. Black means dissolved.' },
        { title: 'Refill', left: 'ndf', right: 'residual',
          text: 'Residual Gaussians fill the dissolved regions and add fine, view-dependent detail.' },
        { title: 'Fuse', left: 'mesh', right: 'ours',
          text: 'Continuous fusion blends both branches into the final image.' },
      ],
    },
    hardz: {
      base: 'assets/hardz/',
      scenes: DATA.hardz || [],
      layers: [
        { id: 'gt', label: 'Ground truth' },
        { id: 'mesh', label: 'Textured mesh' },
        { id: 'hardz', label: 'Hard-Z' },
        { id: 'ours', label: 'Gaussolve' },
      ],
      left: 'hardz', right: 'ours',
    },
  };

  /* ------------------------------------------------------- compare widget */
  function buildCompare(root) {
    const cfg = SETS[root.dataset.set];
    if (!cfg.scenes.length) return;
    const name = root.dataset.name || 'comparison';
    let scene = cfg.scenes[0];
    let viewIdx = 0;
    let left = cfg.left;
    let right = cfg.right;
    let pos = 50;
    let lensOn = false;
    let dragging = false;
    let touched = false;
    let lastPointer = null;

    const labelOf = (id) => cfg.layers.find((l) => l.id === id).label;
    const view = () => scene.views[viewIdx];
    const src = (id, v = view()) => `${cfg.base}${scene.id}/${v.id}/${id}.webp${ASSET_V ? `?v=${ASSET_V}` : ''}`;

    const chips = h('div', { class: 'chips', role: 'group', 'aria-label': 'Scene' });
    cfg.scenes.forEach((s) => {
      const b = h('button', { type: 'button', class: 'chip', 'aria-pressed': 'false' }, s.label);
      b.addEventListener('click', () => { scene = s; viewIdx = 0; render(); });
      chips.append(b);
    });
    const viewSeg = h('div', { class: 'view-pick', role: 'group', 'aria-label': 'View' });
    viewSeg.addEventListener('pointerdown', (e) => e.stopPropagation());
    const top = h('div', { class: 'cmp-top' });
    top.append(chips);

    const mkSelect = (which) => {
      const sel = h('select', { 'aria-label': `${which === 'l' ? 'Left' : 'Right'} image, ${name}` });
      cfg.layers.forEach((l) => sel.append(h('option', { value: l.id }, l.label)));
      sel.addEventListener('change', () => {
        if (which === 'l') left = sel.value; else right = sel.value;
        clearStory();
        render();
      });
      return sel;
    };
    const selL = mkSelect('l');
    const selR = mkSelect('r');
    const swap = h('button', { type: 'button', class: 'icon-btn', 'aria-label': 'Swap left and right' }, ICON_SWAP + 'Swap');
    swap.addEventListener('click', () => { [left, right] = [right, left]; clearStory(); render(); });
    const lensBtn = h('button', { type: 'button', class: 'icon-btn', 'aria-pressed': 'false' }, ICON_LENS + 'Zoom');
    lensBtn.addEventListener('click', () => {
      lensOn = !lensOn;
      lensBtn.setAttribute('aria-pressed', String(lensOn));
      stage.classList.toggle('lens-on', lensOn);
      if (!lensOn) lens.hidden = true;
    });
    const controls = h('div', { class: 'cmp-controls' });
    controls.append(selL, swap, selR, h('span', { class: 'spacer' }), lensBtn);

    const stage = h('div', { class: 'stage' });
    const clip = h('div', { class: 'stage-clip' });
    const imgL = h('img', { class: 'img-l', alt: '', decoding: 'async', draggable: 'false' });
    const imgR = h('img', { class: 'img-r', alt: '', decoding: 'async', draggable: 'false' });
    const tagL = h('span', { class: 'tag tag-l' });
    const tagR = h('span', { class: 'tag tag-r' });
    clip.append(imgL, imgR, tagL, tagR);
    const divider = h('div', { class: 'divider', role: 'slider', tabindex: '0', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-label': `Divider, ${name}` });
    divider.append(h('span', { class: 'knob' }, ICON_ARROWS));
    const lens = h('div', { class: 'lens', hidden: true });
    const lensL = h('div'); const lensR = h('div'); const lensLine = h('div', { class: 'lens-line' });
    lens.append(lensL, lensR, lensLine);
    stage.append(clip, divider, lens, viewSeg);

    let storyBtns = []; let storyText = null;
    // the frame keeps one height for every scene and view, so nothing around it moves
    const frame = h('div', { class: 'stage-frame' });
    frame.append(stage);
    const inner = h('div', { class: 'cmp-inner' });
    inner.append(top, controls, frame);
    root.append(inner);
    // frame height from the typical (median) aspect ratio, so no image gets large margins
    const ars = cfg.scenes.flatMap((s) => s.views.map((v) => v.w / v.h)).sort((a, b) => a - b);
    const arMid = ars[Math.floor(ars.length / 2)];
    let shown = null; // the view whose images are currently on screen
    function layout() {
      const W = frame.clientWidth;
      if (!W) return;
      const fh = Math.round(Math.min(window.innerHeight * 0.72, W / arMid));
      frame.style.height = `${fh}px`;
      const v = shown || view();
      const ar = v.w / v.h;
      const w = Math.min(W, fh * ar);
      stage.style.width = `${Math.round(w)}px`;
      stage.style.height = `${Math.round(w / ar)}px`;
    }
    new ResizeObserver(layout).observe(frame);
    window.addEventListener('resize', layout);
    if (cfg.steps) {
      const story = h('div', { class: 'story', role: 'group', 'aria-label': 'Subtract and refill, step by step' });
      cfg.steps.forEach((st, i) => {
        const b = h('button', { type: 'button', 'aria-pressed': 'false' }, `<b><span class="n">${i + 1}</span>${st.title}</b>`);
        b.addEventListener('click', () => playStep(i));
        storyBtns.push(b);
        story.append(b);
      });
      storyText = h('p', { class: 'story-text', 'aria-live': 'polite' });
      inner.append(story, storyText);
    }

    function clearStory() {
      storyBtns.forEach((b) => b.setAttribute('aria-pressed', 'false'));
      if (storyText) storyText.textContent = '';
    }
    function playStep(i) {
      const st = cfg.steps[i];
      left = st.left; right = st.right;
      storyBtns.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
      storyText.textContent = st.text;
      render();
      sweep([88, 50]);
    }
    let anim = 0;
    function sweep(keys) {
      if (reduceMotion) { setPos(keys[keys.length - 1]); return; }
      const id = ++anim;
      const seg = 650;
      const t0 = performance.now();
      const step = (t) => {
        if (id !== anim) return;
        const k = Math.min(keys.length - 1, (t - t0) / seg);
        const i = Math.min(keys.length - 2, Math.floor(k));
        const f = k - i;
        const e = f < 0.5 ? 4 * f * f * f : 1 - Math.pow(-2 * f + 2, 3) / 2;
        setPos(keys[i] + (keys[i + 1] - keys[i]) * e);
        if (k < keys.length - 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
      setTimeout(() => { if (id === anim) setPos(keys[keys.length - 1]); }, seg * (keys.length - 1) + 80);
    }

    function setPos(p) {
      pos = Math.max(0, Math.min(100, p));
      stage.style.setProperty('--pos', pos + '%');
      divider.setAttribute('aria-valuenow', String(Math.round(pos)));
      divider.setAttribute('aria-valuetext', `${labelOf(left)} ${Math.round(pos)}%, ${labelOf(right)} ${100 - Math.round(pos)}%`);
      tagL.style.opacity = pos < 14 ? '0' : '1';
      tagR.style.opacity = pos > 86 ? '0' : '1';
      if (lensOn && lastPointer) updateLens(lastPointer);
    }

    function render() {
      $$('.chip', chips).forEach((b, i) => b.setAttribute('aria-pressed', String(cfg.scenes[i] === scene)));
      viewSeg.innerHTML = '<span class="vlabel">View</span>';
      scene.views.forEach((v, i) => {
        const b = h('button', { type: 'button', 'aria-pressed': String(i === viewIdx), 'aria-label': `View ${i + 1}` }, String(i + 1));
        b.addEventListener('click', () => { viewIdx = i; render(); });
        viewSeg.append(b);
      });
      viewSeg.hidden = scene.views.length < 2;
      selL.value = left; selR.value = right;
      // load both images first, then swap pictures and geometry in the same frame
      const token = ++renderToken;
      const v = view(); const sL = src(left); const sR = src(right);
      const lbL = labelOf(left); const lbR = labelOf(right); const sceneLabel = scene.label;
      const ready = (u) => new Promise((res) => {
        const im = new Image();
        im.onload = im.onerror = () => (im.decode ? im.decode().catch(() => {}).then(res) : res());
        im.src = u;
      });
      const apply = () => {
        if (token !== renderToken) return;
        shown = v;
        imgL.src = sL; imgR.src = sR;
        imgL.alt = `${sceneLabel}: ${lbL}`;
        imgR.alt = `${sceneLabel}: ${lbR}`;
        tagL.textContent = lbL; tagR.textContent = lbR;
        lensL.style.backgroundImage = `url("${sL}")`;
        lensR.style.backgroundImage = `url("${sR}")`;
        layout();
        setPos(pos);
      };
      // first paint: show at once and let the images stream in; later switches wait until ready
      if (!shown) apply(); else Promise.all([ready(sL), ready(sR)]).then(apply);
      cfg.layers.forEach((l) => { const im = new Image(); im.decoding = 'async'; im.src = src(l.id); });
    }
    let renderToken = 0;

    function pointer(e) {
      const r = clip.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, hgt: r.height };
    }
    function updateLens(p) {
      const { x, y, w, hgt } = p;
      if (x < 0 || y < 0 || x > w || y > hgt) { lens.hidden = true; return; }
      const L = lens.offsetWidth || 210;
      const z = Math.max(2, Math.min(4, (view().w / w) * 1.6));
      const bw = w * z; const bh = hgt * z;
      const bx = L / 2 - x * z; const by = L / 2 - y * z;
      for (const d of [lensL, lensR]) {
        d.style.backgroundSize = `${bw}px ${bh}px`;
        d.style.backgroundPosition = `${bx}px ${by}px`;
      }
      const cut = L / 2 + ((pos / 100) * w - x) * z;
      lensR.style.clipPath = `inset(0 0 0 ${Math.max(0, Math.min(L, cut))}px)`;
      lensLine.style.left = `${cut - 1}px`;
      lensLine.style.display = cut > 0 && cut < L ? 'block' : 'none';
      lens.style.left = `${x - L / 2}px`;
      lens.style.top = `${y - L / 2}px`;
      lens.hidden = false;
    }

    stage.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      dragging = true; touched = true; anim++;
      stage.setPointerCapture(e.pointerId);
      const p = pointer(e); lastPointer = p;
      setPos((p.x / p.w) * 100);
    });
    stage.addEventListener('pointermove', (e) => {
      const p = pointer(e); lastPointer = p;
      if (dragging) setPos((p.x / p.w) * 100);
      if (lensOn && e.pointerType !== 'touch') updateLens(p);
    });
    const end = () => { dragging = false; };
    stage.addEventListener('pointerup', end);
    stage.addEventListener('pointercancel', end);
    stage.addEventListener('pointerleave', () => { if (!dragging) lens.hidden = true; });
    divider.addEventListener('keydown', (e) => {
      const big = e.shiftKey ? 10 : 2;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') setPos(pos - big);
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') setPos(pos + big);
      else if (e.key === 'Home') setPos(0);
      else if (e.key === 'End') setPos(100);
      else return;
      touched = true; anim++;
      e.preventDefault();
    });

    // the first time the viewer scrolls into view, sweep the divider once to show it can be dragged
    const io = new IntersectionObserver((ents) => {
      if (ents.some((en) => en.isIntersecting)) {
        io.disconnect();
        if (!touched) setTimeout(() => { if (!touched) sweep([50, 82, 22, 50]); }, 250);
      }
    }, { threshold: 0.6 });
    io.observe(stage);
    render();
  }

  /* ------------------------------------------------- view-dependent dissolve */
  function buildSH(root) {
    const groups = ['g1', 'g2', 'g3'];
    const parts = [
      { id: 'dissolve', label: 'Dissolve field' },
      { id: 'residual', label: 'Gaussians' },
      { id: 'blend', label: 'Fused' },
    ];
    let variant = 'sh';
    const names = { sh: 'with scalar SH', nosh: 'without SH' };

    const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Dissolve field' });
    const bSH = h('button', { type: 'button', 'aria-pressed': 'true' }, 'With scalar SH');
    const bNo = h('button', { type: 'button', 'aria-pressed': 'false' }, 'Without SH');
    seg.append(bSH, bNo);
    const bar = h('div', { class: 'toolbar' });
    bar.append(seg);
    const head = h('div', { class: 'sh-head', 'aria-hidden': 'true' });
    parts.forEach((p) => head.append(h('span', {}, p.label)));

    const grid = h('div', { class: 'sh-grid' });
    const imgs = [];
    groups.forEach((g) => {
      const row = h('div', { class: 'sh-row' });
      parts.forEach((p) => {
        const cell = h('div', { class: 'sh-cell' });
        const im = h('img', { alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' });
        im.dataset.g = g; im.dataset.p = p.id; im.dataset.label = p.label;
        cell.append(im);
        const flip = (on) => setImgs(on ? (variant === 'sh' ? 'nosh' : 'sh') : variant, cell);
        cell.addEventListener('pointerdown', (e) => { e.preventDefault(); flip(true); });
        ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) => cell.addEventListener(ev, () => flip(false)));
        row.append(cell);
        imgs.push(im);
      });
      grid.append(row);
    });
    root.append(bar, head, grid);

    function setImgs(v, onlyCell) {
      imgs.forEach((im) => {
        if (onlyCell && !onlyCell.contains(im)) return;
        im.src = `assets/sh/${im.dataset.g}/${v}_${im.dataset.p}.webp`;
        im.alt = `${im.dataset.label}, ${names[v]}`;
      });
    }
    function set(v) {
      variant = v;
      bSH.setAttribute('aria-pressed', String(v === 'sh'));
      bNo.setAttribute('aria-pressed', String(v === 'nosh'));
      setImgs(v);
    }
    bSH.addEventListener('click', () => set('sh'));
    bNo.addEventListener('click', () => set('nosh'));
    groups.forEach((g) => parts.forEach((p) => ['sh', 'nosh'].forEach((v) => { const im = new Image(); im.src = `assets/sh/${g}/${v}_${p.id}.webp`; })));
    set('sh');
  }

  /* ------------------------------------------- results: chart + main table */
  // dgs = Gaussian reduction vs 3DGS (%)
  const RESULTS = {
    hypersim: {
      label: 'ML-Hypersim-AI',
      rows: [
        { m: '3DGS', fam: 'gs', mesh: 0, hyb: 0, psnr: 34.39, ssim: 0.959, lpips: 0.155, dgs: 0 },
        { m: 'Reduced 3DGS', fam: 'gs', mesh: 0, hyb: 0, psnr: 34.04, ssim: 0.957, lpips: 0.153, dgs: 62 },
        { m: '3DGS-MCMC', fam: 'gs', mesh: 0, hyb: 0, psnr: 35.82, ssim: 0.960, lpips: 0.140, dgs: -1 },
        { m: 'SuGaR', fam: 'surf', mesh: 1, hyb: 0, psnr: 32.29, ssim: 0.950, lpips: 0.149, dgs: 33 },
        { m: 'PGSR', fam: 'surf', mesh: 1, hyb: 0, psnr: 32.95, ssim: 0.958, lpips: 0.147, dgs: 61 },
        { m: 'MeshSplatting', fam: 'surf', mesh: 1, hyb: 0, psnr: 28.47, ssim: 0.903, lpips: 0.236, dgs: null },
        { m: '3D-GES', fam: 'hyb', mesh: 0, hyb: 1, psnr: 34.87, ssim: 0.961, lpips: 0.130, dgs: 63 },
        { m: 'Hard-Z Proxy', fam: 'hz', mesh: 1, hyb: 1, psnr: 34.31, ssim: 0.958, lpips: 0.119, dgs: 27 },
        { m: 'Ours (3DGS)', fam: 'ours', from: '3DGS', mesh: 1, hyb: 1, psnr: 35.35, ssim: 0.962, lpips: 0.108, dgs: 30 },
        { m: 'Ours (Reduced 3DGS)', fam: 'ours', from: 'Reduced 3DGS', mesh: 1, hyb: 1, psnr: 35.29, ssim: 0.959, lpips: 0.113, dgs: 67 },
        { m: 'Ours (3DGS-MCMC)', fam: 'ours', from: '3DGS-MCMC', mesh: 1, hyb: 1, psnr: 36.89, ssim: 0.961, lpips: 0.110, dgs: -6 },
      ],
    },
    deepblending: {
      label: 'Deep Blending',
      rows: [
        { m: '3DGS', fam: 'gs', mesh: 0, hyb: 0, psnr: 28.50, ssim: 0.877, lpips: 0.272, dgs: 0 },
        { m: 'Reduced 3DGS', fam: 'gs', mesh: 0, hyb: 0, psnr: 28.43, ssim: 0.875, lpips: 0.277, dgs: 51 },
        { m: '3DGS-MCMC', fam: 'gs', mesh: 0, hyb: 0, psnr: 28.56, ssim: 0.875, lpips: 0.268, dgs: -12 },
        { m: 'SuGaR', fam: 'surf', mesh: 1, hyb: 0, psnr: 27.58, ssim: 0.858, lpips: 0.306, dgs: 19 },
        { m: 'PGSR', fam: 'surf', mesh: 1, hyb: 0, psnr: 28.44, ssim: 0.863, lpips: 0.302, dgs: 28 },
        { m: 'MeshSplatting', fam: 'surf', mesh: 1, hyb: 0, psnr: 24.71, ssim: 0.809, lpips: 0.359, dgs: null },
        { m: '3D-GES', fam: 'hyb', mesh: 0, hyb: 1, psnr: 28.42, ssim: 0.875, lpips: 0.262, dgs: 41 },
        { m: 'Hard-Z Proxy', fam: 'hz', mesh: 1, hyb: 1, psnr: 28.32, ssim: 0.872, lpips: 0.234, dgs: 39 },
        { m: 'Ours (3DGS)', fam: 'ours', from: '3DGS', mesh: 1, hyb: 1, psnr: 28.75, ssim: 0.876, lpips: 0.228, dgs: 47 },
        { m: 'Ours (Reduced 3DGS)', fam: 'ours', from: 'Reduced 3DGS', mesh: 1, hyb: 1, psnr: 28.67, ssim: 0.877, lpips: 0.232, dgs: 65 },
        { m: 'Ours (3DGS-MCMC)', fam: 'ours', from: '3DGS-MCMC', mesh: 1, hyb: 1, psnr: 28.93, ssim: 0.880, lpips: 0.227, dgs: -3 },
      ],
    },
  };
  const METRICS = {
    psnr: { label: 'PSNR (dB)', dir: 1, fmt: (v) => v.toFixed(2) },
    ssim: { label: 'SSIM', dir: 1, fmt: (v) => v.toFixed(3) },
    lpips: { label: 'LPIPS', dir: -1, fmt: (v) => v.toFixed(3) },
  };

  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= n) || 10 * mag;
    const t0 = Math.ceil(lo / step) * step;
    const out = [];
    for (let v = t0; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }

  function buildResults(root) {
    const chartBox = $('.chart-box', root);
    const tip = h('div', { class: 'tooltip', role: 'status' });
    chartBox.append(tip);
    const tbody = $('#main-table tbody', root);
    const dsBtns = $$('[data-ds]', root);
    const mBtns = $$('[data-metric]', root);
    let ds = 'deepblending';
    let metric = 'lpips';

    dsBtns.forEach((b) => b.addEventListener('click', () => { ds = b.dataset.ds; update(); }));
    mBtns.forEach((b) => b.addEventListener('click', () => { metric = b.dataset.metric; update(); }));
    new ResizeObserver(() => drawChart()).observe(chartBox);

    function update() {
      dsBtns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.ds === ds)));
      mBtns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.metric === metric)));
      drawChart();
      drawTable();
    }

    function drawTable() {
      const rows = RESULTS[ds].rows;
      const best = {};
      for (const k of ['psnr', 'ssim', 'lpips', 'dgs']) {
        const vals = rows.map((r) => r[k]).filter((v) => v != null);
        best[k] = k === 'lpips' ? Math.min(...vals) : Math.max(...vals);
      }
      const cell = (r, k, f) => {
        const v = r[k];
        if (v == null) return '<td>–</td>';
        return `<td class="${v === best[k] ? 'best' : ''}">${f(v)}</td>`;
      };
      let prev = null;
      tbody.innerHTML = rows.map((r) => {
        const famKey = r.fam === 'ours' || r.fam === 'hz' ? 'mine' : r.fam;
        const sep = prev && prev !== famKey ? ' sep' : '';
        prev = famKey;
        return `<tr class="${r.fam === 'ours' ? 'ours' : ''}${sep}"><td>${r.m}</td><td>${r.mesh ? '✓' : '–'}</td><td>${r.hyb ? '✓' : '–'}</td>` +
          cell(r, 'psnr', (v) => v.toFixed(2)) + cell(r, 'ssim', (v) => v.toFixed(3)) + cell(r, 'lpips', (v) => v.toFixed(3)) +
          cell(r, 'dgs', (v) => `${v}%`) + '</tr>';
      }).join('');
    }

    function drawChart() {
      const W = Math.max(320, chartBox.clientWidth);
      const narrow = W < 560;
      const H = narrow ? 360 : Math.min(460, Math.max(380, W * 0.36));
      const M = { t: 18, r: narrow ? 14 : 28, b: 48, l: narrow ? 62 : 58 };
      const short = (m) => (narrow ? m.replace('Reduced 3DGS', 'Reduced').replace('3DGS-MCMC', 'MCMC').replace(' Proxy', '') : m);
      const rows = RESULTS[ds].rows.filter((r) => r.dgs != null);
      const met = METRICS[metric];
      const xs = rows.map((r) => r.dgs);
      const ys = rows.map((r) => r[metric]);
      const xlo = Math.min(-15, Math.floor(Math.min(...xs) / 10) * 10);
      const xhi = Math.max(80, Math.ceil(Math.max(...xs) / 10) * 10);
      const ypad = (Math.max(...ys) - Math.min(...ys)) * 0.12 || 0.01;
      const ylo = Math.min(...ys) - ypad;
      const yhi = Math.max(...ys) + ypad;
      const X = (v) => M.l + ((v - xlo) / (xhi - xlo)) * (W - M.l - M.r);
      const Y = (v) => H - M.b - ((v - ylo) / (yhi - ylo)) * (H - M.t - M.b);
      const xt = niceTicks(xlo, xhi, narrow ? 5 : 8);
      const yt = niceTicks(ylo, yhi, 5);
      let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${met.label} versus Gaussian reduction, ${RESULTS[ds].label}">`;
      s += '<defs><marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1.5 L9 5 L0 8.5 z" fill="var(--accent)" opacity=".55"/></marker></defs>';
      yt.forEach((v) => {
        s += `<line class="gridline" x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}"/>`;
        s += `<text class="tick" x="${M.l - 8}" y="${Y(v) + 4}" text-anchor="end">${met.fmt(v)}</text>`;
      });
      xt.forEach((v) => { s += `<text class="tick" x="${X(v)}" y="${H - M.b + 18}" text-anchor="middle">${v}%</text>`; });
      s += `<line class="axis" x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}"/>`;
      if (xlo < 0) s += `<line class="axis" x1="${X(0)}" x2="${X(0)}" y1="${M.t}" y2="${H - M.b}" stroke-dasharray="3 4"/>`;
      s += `<text class="axis-title" x="${(M.l + W - M.r) / 2}" y="${H - 8}" text-anchor="middle">${narrow ? 'Fewer Gaussians than 3DGS →' : 'Fewer Gaussians than 3DGS (ΔGS) →'}</text>`;
      s += `<text class="axis-title" transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${met.label} ${met.dir > 0 ? '↑' : '↓'} better</text>`;
      rows.filter((r) => r.from).forEach((r) => {
        const b = rows.find((q) => q.m === r.from);
        if (!b) return;
        const x1 = X(b.dgs); const y1 = Y(b[metric]); const x2 = X(r.dgs); const y2 = Y(r[metric]);
        const d = Math.hypot(x2 - x1, y2 - y1);
        if (d < 14) return;
        const k1 = 7 / d; const k2 = (d - 9) / d;
        s += `<line class="link" x1="${x1 + (x2 - x1) * k1}" y1="${y1 + (y2 - y1) * k1}" x2="${x1 + (x2 - x1) * k2}" y2="${y1 + (y2 - y1) * k2}" marker-end="url(#arr)"/>`;
      });
      const placed = [];
      const pts = rows.map((r) => ({ r, x: X(r.dgs), y: Y(r[metric]) }));
      pts.forEach((p) => placed.push({ x0: p.x - 6, x1: p.x + 6, y0: p.y - 6, y1: p.y + 6 }));
      let labels = '';
      [...pts].sort((a, b) => (b.r.fam === 'ours') - (a.r.fam === 'ours')).forEach((p) => {
        const txt = short(p.r.m);
        const w = txt.length * (p.r.fam === 'ours' ? 6.9 : 6.4) + 4;
        const cands = [
          [p.x + 9, p.y + 4, 'start'], [p.x - 9, p.y + 4, 'end'],
          [p.x, p.y - 11, 'middle'], [p.x, p.y + 19, 'middle'],
          [p.x + 8, p.y - 9, 'start'], [p.x - 8, p.y - 9, 'end'],
          [p.x + 8, p.y + 17, 'start'], [p.x - 8, p.y + 17, 'end'],
        ];
        for (const [lx, ly, a] of cands) {
          const x0 = a === 'start' ? lx : a === 'end' ? lx - w : lx - w / 2;
          const box = { x0, x1: x0 + w, y0: ly - 11, y1: ly + 3 };
          if (box.x0 < M.l - 4 || box.x1 > W - 2 || box.y0 < 0 || box.y1 > H - M.b - 2) continue;
          if (placed.some((q) => !(box.x1 < q.x0 || box.x0 > q.x1 || box.y1 < q.y0 || box.y0 > q.y1))) continue;
          placed.push(box);
          labels += `<text class="lbl${p.r.fam === 'ours' ? ' ours' : ''}" x="${lx}" y="${ly}" text-anchor="${a}">${txt}</text>`;
          break;
        }
      });
      pts.forEach((p, i) => {
        s += `<g class="pt" data-i="${i}">`;
        s += p.r.fam === 'ours'
          ? `<circle class="pt-ours" cx="${p.x}" cy="${p.y}" r="6.5"/>`
          : `<circle class="pt-base" cx="${p.x}" cy="${p.y}" r="5"/>`;
        s += `<circle class="pt-hit" cx="${p.x}" cy="${p.y}" r="14" tabindex="0" aria-label="${p.r.m}: ${met.label} ${met.fmt(p.r[metric])}, ${p.r.dgs}% fewer Gaussians"/>`;
        s += '</g>';
      });
      s += labels + '</svg>';
      chartBox.querySelector('svg')?.remove();
      chartBox.insertAdjacentHTML('afterbegin', s);
      const svg = chartBox.querySelector('svg');
      const show = (i) => {
        const p = pts[i];
        $$('.pt', svg).forEach((g) => g.classList.toggle('hl', g.dataset.i === String(i)));
        tip.innerHTML = `<b>${p.r.m}</b><table>` +
          `<tr><td>PSNR</td><td>${p.r.psnr.toFixed(2)}</td></tr>` +
          `<tr><td>SSIM</td><td>${p.r.ssim.toFixed(3)}</td></tr>` +
          `<tr><td>LPIPS</td><td>${p.r.lpips.toFixed(3)}</td></tr>` +
          `<tr><td>ΔGS</td><td>${p.r.dgs}%</td></tr></table>`;
        const sx = svg.clientWidth / W;
        tip.style.left = Math.max(92, Math.min(chartBox.clientWidth - 92, p.x * sx)) + 'px';
        tip.style.top = p.y * sx + 'px';
        tip.classList.add('show');
      };
      const hide = () => { tip.classList.remove('show'); $$('.pt', svg).forEach((g) => g.classList.remove('hl')); };
      $$('.pt-hit', svg).forEach((c) => {
        const i = +c.parentNode.dataset.i;
        c.addEventListener('pointerenter', () => show(i));
        c.addEventListener('pointerleave', hide);
        c.addEventListener('focus', () => show(i));
        c.addEventListener('blur', hide);
        c.addEventListener('click', () => show(i));
      });
    }
    update();
  }

  /* --------------------------------------------------------------- misc */
  function initCopy() {
    const btn = $('#copy-bib');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const text = $('#bibtex').textContent;
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        const r = document.createRange(); r.selectNodeContents($('#bibtex'));
        const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
        document.execCommand('copy'); sel.removeAllRanges();
      }
      const old = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = old; }, 1600);
    });
  }
  function initNav() {
    const links = $$('.nav-links a');
    const secs = links.map((a) => [a, document.getElementById(a.getAttribute('href').slice(1))]).filter(([, el]) => el);
    let ticking = false;
    const update = () => {
      ticking = false;
      const y = window.innerHeight * 0.35;
      let cur = null;
      secs.forEach(([a, el]) => { if (el.getBoundingClientRect().top <= y) cur = a; });
      links.forEach((a) => a.classList.toggle('active', a === cur));
    };
    window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  }

  $$('.cmp[data-set]').forEach(buildCompare);
  $$('.sh[data-set]').forEach(buildSH);
  const res = $('#results-interactive');
  if (res) buildResults(res);
  initCopy();
  initNav();
})();
