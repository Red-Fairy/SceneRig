// Material passes share a trajectory; the VIGA inset follows the same simulation clock.
(() => {
  const find = (id) => document.getElementById(id);
  const videos = [find('simulation-rgb'), find('simulation-clay'), find('simulation-viga')];
  const [master] = videos;
  const followers = videos.slice(1);
  const inset = find('simulation-inset');
  const closeInset = find('simulation-inset-close');
  const showInset = find('simulation-inset-show');
  let insetDismissed = false;
  const source = find('simulation-source');
  const play = find('simulation-play');
  const restart = find('simulation-restart');
  const scrub = find('simulation-time');
  const status = find('simulation-status');
  const state = find('simulation-state');
  const clock = find('simulation-clock');
  const note = find('simulation-note');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  source.disabled = true;
  let wanted = !reduced.matches;
  let visible = false;
  let ready = false;
  let generation = 0;
  let frame = 0;
  let manifest;
  let waiting = false;
  let starting = false;
  const duration = () => manifest?.seconds || 5;
  const active = () => ready && wanted && visible && !document.hidden;

  function updateInset() {
    const available = source.value !== 'viga';
    inset.hidden = !available || insetDismissed;
    showInset.hidden = !available || !insetDismissed;
    showInset.setAttribute('aria-expanded', String(!inset.hidden));
  }
  closeInset.addEventListener('click', () => {
    insetDismissed = true;
    updateInset();
    showInset.focus({ preventScroll: true });
  });
  showInset.addEventListener('click', () => {
    insetDismissed = false;
    updateInset();
    closeInset.focus({ preventScroll: true });
  });

  function update() {
    const running = active() && videos.every(v => !v.paused && v.readyState >= 3);
    status.classList.toggle('playing', running);
    if (ready) state.textContent = running ? 'Playing' : waiting ? 'Loading' : 'Paused';
    play.innerHTML = `<span aria-hidden="true">${wanted ? 'Ⅱ' : '▶'}</span>`;
    play.setAttribute('aria-label', wanted ? 'Pause simulation' : 'Play simulation');
    play.setAttribute('aria-pressed', String(wanted));
    const time = Math.min(master.currentTime || 0, duration());
    scrub.value = String(time);
    scrub.setAttribute('aria-valuetext', `${time.toFixed(1)} of ${duration().toFixed(1)} seconds`);
    clock.textContent = `${time.toFixed(1)} / ${duration().toFixed(1)} s`;
  }
  function pause() {
    videos.forEach(v => v.pause());
    cancelAnimationFrame(frame);
    update();
  }
  function seek(time) {
    videos.forEach(v => { v.currentTime = Math.max(0, Math.min(time, duration())); });
    update();
  }
  async function reconcile() {
    if (!active()) { pause(); return; }
    if (starting) return;
    const request = generation;
    starting = true;
    if (master.ended || master.currentTime >= duration()) seek(0);
    followers.forEach(v => {
      if (Math.abs(master.currentTime - v.currentTime) > .06) v.currentTime = master.currentTime;
    });
    try {
      await Promise.all(videos.map(v => v.play()));
      if (request !== generation) return;
      if (!active()) { pause(); return; }
      waiting = false;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(tick);
    } catch (error) {
      if (request !== generation || error.name === 'AbortError') return;
      wanted = false;
      pause();
      note.textContent = 'Press play to start the recorded simulation.';
    } finally {
      starting = false;
      update();
      if (request !== generation && active()) reconcile();
    }
  }
  function tick() {
    if (!active()) { pause(); return; }
    // A delayed decoder must not leave one material at a different physical state.
    if (!videos.some(v => v.seeking)) followers.forEach(v => {
      if (Math.abs(master.currentTime - v.currentTime) > .08) v.currentTime = master.currentTime;
    });
    update();
    frame = requestAnimationFrame(tick);
  }
  function fail() {
    ready = false;
    wanted = false;
    pause();
    state.textContent = 'Unavailable';
    play.disabled = restart.disabled = scrub.disabled = true;
    note.textContent = 'This recording could not load. Choose another method or refresh to retry.';
  }
  function load() {
    const method = manifest.methods[source.value];
    updateInset();
    generation++;
    ready = false;
    waiting = false;
    pause();
    state.textContent = 'Loading';
    play.disabled = restart.disabled = scrub.disabled = true;
    note.textContent = method.note || 'Recorded physics · 1× speed · drag the divider to compare materials.';
    videos.forEach((v, i) => {
      const recording = i === 2 ? manifest.methods.viga : method;
      const material = i === 1 ? 'clay' : 'rgb';
      v.poster = recording[`${material}_poster`];
      v.setAttribute('aria-label', `${material === 'rgb' ? 'RGB' : 'Clay'} view of the recorded ${recording.label} gravity simulation`);
      v.src = recording[material];
      v.load();
    });
    update();
  }
  videos.forEach(v => {
    v.addEventListener('error', fail);
    v.addEventListener('canplay', () => {
      if (!videos.every(video => video.readyState >= 3)) return;
      ready = true;
      waiting = false;
      play.disabled = restart.disabled = scrub.disabled = false;
      reconcile();
    });
    v.addEventListener('waiting', () => {
      if (!ready || !active() || v.seeking) return;
      waiting = true;
      pause();
    });
    v.addEventListener('seeked', () => {
      if (!videos.some(video => video.seeking)) reconcile();
    });
    v.addEventListener('timeupdate', update);
  });
  master.addEventListener('ended', () => {
    pause();
    if (wanted) { seek(0); reconcile(); }
  });
  play.addEventListener('click', () => { wanted = !wanted; reconcile(); });
  restart.addEventListener('click', () => { pause(); seek(0); reconcile(); });
  scrub.addEventListener('input', () => {
    const time = Number(scrub.value);
    wanted = false;
    pause();
    seek(time);
  });
  source.addEventListener('change', load);
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    reconcile();
  }, { threshold: .15 }).observe(find('clay-slider'));
  document.addEventListener('visibilitychange', reconcile);
  reduced.addEventListener('change', () => {
    if (reduced.matches) wanted = false;
    reconcile();
  });
  fetch('data/simulation.json', { cache: 'no-cache' })
    .then(response => { if (!response.ok) throw new Error('Missing simulation manifest'); return response.json(); })
    .then(data => { manifest = data; source.disabled = false; scrub.max = String(duration()); load(); })
    .catch(fail);
})();
