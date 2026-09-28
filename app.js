const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );

async function readData(name) {
  const response = await fetch(`data/${name}.json`, { cache: "no-cache" });
  if (!response.ok) throw new Error(`Cannot load ${name}`);
  return response.json();
}

// The reveal uses the same render dimensions and camera for both materials.
function setupClaySlider() {
  const slider = $("#clay-slider");
  const range = $("#clay-range");
  const button = $("#clay-motion");
  let enabled = !reducedMotion.matches;
  let visible = false;
  let frame = 0;
  let previous = 0;
  let phase = Math.asin((Number(range.value) - 50) / 35);

  function setPosition(value) {
    range.value = String(Math.round(value));
    range.setAttribute(
      "aria-valuetext",
      `${Math.round(value)} percent textured`,
    );
    slider.style.setProperty("--split", `${value}%`);
  }
  function updateButton() {
    button.textContent = enabled ? "Pause divider" : "Animate divider";
    button.setAttribute("aria-pressed", String(enabled));
  }
  function animate(time) {
    if (previous) phase += Math.min(time - previous, 50) / 2200;
    previous = time;
    setPosition(50 + Math.sin(phase) * 35);
    frame = requestAnimationFrame(animate);
  }
  function updateAnimation() {
    cancelAnimationFrame(frame);
    previous = 0;
    if (enabled && visible && !document.hidden)
      frame = requestAnimationFrame(animate);
    updateButton();
  }
  range.addEventListener("input", () => {
    enabled = false;
    setPosition(Number(range.value));
    updateAnimation();
  });
  button.addEventListener("click", () => {
    enabled = !enabled;
    phase = Math.asin(
      Math.max(-1, Math.min(1, (Number(range.value) - 50) / 35)),
    );
    updateAnimation();
  });
  new IntersectionObserver(
    (entries) => {
      visible = entries[0].isIntersecting;
      updateAnimation();
    },
    { threshold: 0.1 },
  ).observe(slider);
  document.addEventListener("visibilitychange", updateAnimation);
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) enabled = false;
    updateAnimation();
  });
  updateButton();
}

function setupGallery(data) {
  let groupId = data.groups[0].id;
  let viewId = "source";
  let category = "all";
  let selectedId = data.scenes[0].sourceId;
  const sceneSelect = $("#scene-select");
  const categorySelect = $("#category-select");
  const groups = $("#model-groups");
  const views = $("#gallery-views");
  const sceneIds = [...new Set(data.scenes.map((scene) => scene.sourceId))];
  const sceneLabel = (scene) =>
    `Scene ${String(sceneIds.indexOf(scene.sourceId) + 1).padStart(2, "0")}`;

  groups.innerHTML = data.groups
    .map(
      (group) =>
        `<button data-group="${group.id}" aria-pressed="false">${escapeHTML(group.label)}</button>`,
    )
    .join("");
  views.innerHTML = data.views
    .map(
      (view) =>
        `<button data-view="${view.id}" aria-pressed="false">${escapeHTML(view.label)}</button>`,
    )
    .join("");
  const categories = [...new Set(data.scenes.map((scene) => scene.category))];
  categorySelect.innerHTML += categories
    .map(
      (name) =>
        `<option value="${escapeHTML(name)}">${escapeHTML(name)}</option>`,
    )
    .join("");
  const filteredScenes = () =>
    data.scenes.filter(
      (scene) =>
        scene.group === groupId &&
        (category === "all" || scene.category === category),
    );

  function comparisonCard(image, label, type, subtitle) {
    return `<figure class="comparison-card ${type}"><img class="comparison-image" src="${image}" width="720" height="450" alt="${escapeHTML(label)}${subtitle ? `: ${escapeHTML(subtitle)}` : ""}"><figcaption>${escapeHTML(label)}${subtitle ? `<small>${escapeHTML(subtitle)}</small>` : ""}</figcaption></figure>`;
  }
  function renderScene() {
    const scenes = filteredScenes();
    const scene =
      scenes.find((item) => item.sourceId === selectedId) || scenes[0];
    selectedId = scene.sourceId;
    sceneSelect.value = selectedId;
    const group = data.groups.find((item) => item.id === groupId);
    const viewLabel = viewId === "source" ? "Source view" : "Nearby view";
    // Put SceneRig immediately after the input in both comparison groups.
    const order = ["ours", "viga", "rest3d", "sc", "simfoundry", "codex"];
    const methods = [...scene.methods].sort(
      (a, b) => order.indexOf(a.id) - order.indexOf(b.id),
    );
    $("#comparison-grid").innerHTML =
      comparisonCard(scene.input, "Input image", "input", "") +
      methods
        .map((method) =>
          comparisonCard(method[viewId], method.label, method.id, viewLabel),
        )
        .join("");
    $("#comparison-grid").classList.toggle("four-panels", methods.length === 3);
    $("#gallery-note").textContent = group.note;
    $("#scene-count").textContent =
      `${scenes.indexOf(scene) + 1} / ${scenes.length}`;
    $("#gallery-total").textContent =
      `${scenes.length} scenes · ${group.methods.length} methods`;
    $$("[data-group]", groups).forEach((button) =>
      setActive(button, button.dataset.group === groupId),
    );
    $$("[data-view]", views).forEach((button) =>
      setActive(button, button.dataset.view === viewId),
    );
    $$(".scene-thumb").forEach((button) =>
      setActive(button, button.dataset.scene === selectedId),
    );
  }
  function renderCollection() {
    const scenes = filteredScenes();
    sceneSelect.innerHTML = scenes
      .map(
        (scene) =>
          `<option value="${scene.sourceId}">${sceneLabel(scene)}</option>`,
      )
      .join("");
    $("#scene-thumbnails").innerHTML = scenes
      .map(
        (scene) =>
          `<button class="scene-thumb" data-scene="${scene.sourceId}" aria-label="View ${sceneLabel(scene)}" aria-pressed="false"><img src="${scene.input}" alt="" loading="lazy" width="192" height="108"></button>`,
      )
      .join("");
    renderScene();
  }
  groups.addEventListener("click", (event) => {
    const button = event.target.closest("[data-group]");
    if (!button) return;
    groupId = button.dataset.group;
    renderCollection();
  });
  views.addEventListener("click", (event) => {
    const button = event.target.closest("[data-view]");
    if (!button) return;
    viewId = button.dataset.view;
    renderScene();
  });
  categorySelect.addEventListener("change", () => {
    category = categorySelect.value;
    renderCollection();
  });
  sceneSelect.addEventListener("change", () => {
    selectedId = sceneSelect.value;
    renderScene();
  });
  $("#scene-thumbnails").addEventListener("click", (event) => {
    const button = event.target.closest("[data-scene]");
    if (!button) return;
    selectedId = button.dataset.scene;
    renderScene();
  });
  function stepScene(direction) {
    const scenes = filteredScenes();
    const index = scenes.findIndex((scene) => scene.sourceId === selectedId);
    selectedId =
      scenes[(index + direction + scenes.length) % scenes.length].sourceId;
    renderScene();
  }
  $("#scene-prev").addEventListener("click", () => stepScene(-1));
  $("#scene-next").addEventListener("click", () => stepScene(1));
  renderCollection();
}

function setActive(button, active) {
  button.classList.toggle("active", active);
  button.setAttribute("aria-pressed", String(active));
}

function outcomeHTML(outcome) {
  return `<span class="video-outcome${outcome.success ? "" : " failure"}">${escapeHTML(outcome.label)}</span>`;
}
function videoHTML(media, label, controls = true) {
  return `<video ${controls ? "controls" : ""} muted playsinline preload="none" poster="${media.start_poster}" aria-label="${escapeHTML(label)}"><source src="${media.src}" type="video/mp4">Your browser cannot play this video. <a href="${media.src}">Download the recording.</a></video>`;
}

// Replay has a shared time axis. Policy videos retain independent clocks.
function bindPlayback(container, button, synchronized, playbackRate = 1, startTime = 0, startTogetherOnce = false) {
  const videos = $$("video", container);
  let syncing = false;
  let groupStarted = false;
  let disposed = false;
  const events = new AbortController();
  function updateButton() {
    button.textContent = videos.some((video) => !video.paused && !video.ended)
      ? "Ⅱ Pause all"
      : "▶ Play all";
  }
  async function play() {
    groupStarted = true;
    const results = await Promise.allSettled(
      videos.filter((video) => !video.ended).map((video) => video.play()),
    );
    if (disposed) return;
    const failed = results.some((result) => result.status === "rejected");
    button.title = failed
      ? "Use the individual video controls if playback is blocked by your browser."
      : "";
    updateButton();
  }
  function pause() {
    videos.forEach((video) => video.pause());
    updateButton();
  }
  function restart() {
    videos.forEach((video) => {
      video.currentTime = startTime;
    });
    return play();
  }
  function alignTo(source) {
    if (syncing) return;
    syncing = true;
    videos
      .filter((video) => video !== source && video.readyState > 0)
      .forEach((video) => {
        if (Math.abs(video.currentTime - source.currentTime) > 0.15)
          video.currentTime = source.currentTime;
      });
    syncing = false;
  }
  for (const video of videos) {
    video.muted = true;
    video.defaultPlaybackRate = playbackRate;
    video.playbackRate = playbackRate;
    if (startTime) video.currentTime = startTime;
    video.addEventListener(
      "play",
      () => {
        // Loading queues media events; honor a newer pause before synchronizing.
        if (video.paused) return;
        if (synchronized || (startTogetherOnce && !groupStarted)) {
          groupStarted = true;
          if (synchronized) alignTo(video);
          videos
            .filter((other) => other !== video && other.paused && !other.ended)
            .forEach((other) => other.play().catch(() => {}));
        }
        updateButton();
      },
      { signal: events.signal },
    );
    video.addEventListener(
      "pause",
      () => {
        if (!video.paused) return;
        if (synchronized)
          videos
            .filter((other) => other !== video && !other.paused)
            .forEach((other) => other.pause());
        updateButton();
      },
      { signal: events.signal },
    );
    video.addEventListener("ended", updateButton, { signal: events.signal });
    if (synchronized) {
      video.addEventListener("seeking", () => alignTo(video), {
        signal: events.signal,
      });
      video.addEventListener(
        "ratechange",
        () =>
          videos.forEach((other) => {
            if (other.playbackRate !== video.playbackRate)
              other.playbackRate = video.playbackRate;
          }),
        { signal: events.signal },
      );
    }
  }
  const timer = synchronized
    ? setInterval(() => {
        const leader = videos[0];
        if (!leader.paused && !leader.seeking && leader.readyState >= 3)
          alignTo(leader);
      }, 900)
    : null;
  button.addEventListener(
    "click",
    () => {
      if (videos.some((video) => !video.paused && !video.ended)) pause();
      else if (videos.every((video) => video.ended)) restart();
      else play();
    },
    { signal: events.signal },
  );
  return {
    videos,
    play,
    pause,
    restart,
    dispose() {
      disposed = true;
      events.abort();
      if (timer) clearInterval(timer);
      videos.forEach((video) => video.pause());
    },
  };
}

function setupHeroVideos(data) {
  const selections = [
    {
      id: "policy-01",
      title: "Real-to-sim policy evaluation",
      task: "Put mustard bottle in the left bin",
      note: "Same policy · independent executions",
    },
    {
      id: "replay-01",
      title: "Open-loop trajectory replay",
      task: "Put everything into the bin",
      note: "Same recorded commands · shared timeline",
    },
  ];
  $("#hero-robotics").innerHTML = selections
    .map((selection) => {
      const item = data.items.find((episode) => episode.id === selection.id);
      const note = `${selection.note} · ${data.playback_rates[item.kind]}× speed${item.presentation_note ? ` · ${item.presentation_note}` : ""}`;
      return `<article class="hero-application" data-hero="${selection.id}"><header><div><h3>${selection.title}</h3><p class="hero-task">Task: ${escapeHTML(selection.task)}</p><p>${escapeHTML(note)}</p></div><button class="hero-play" type="button">▶ Play all</button></header><div class="hero-video-row">${["real", "baseline", "ours"].map((role) => `<figure data-role="${role}">${videoHTML(item.media[role], `${role === "real" ? "Real episode" : data.methods[role]} — ${selection.title}: ${selection.task}`, true)}<figcaption><span>${role === "real" ? "Real episode" : data.methods[role]}</span>${outcomeHTML(item.outcomes[role])}</figcaption></figure>`).join("")}</div></article>`;
    })
    .join("");
  $$(".hero-application").forEach((container) => {
    const item = data.items.find(
      (episode) => episode.id === container.dataset.hero,
    );
    const button = $(".hero-play", container);
    const playback = bindPlayback(
      container,
      button,
      item.synchronized,
      data.playback_rates[item.kind],
      item.kind === "replay" ? 12 : 0,
      item.kind === "policy",
    );
    // Visibility can pause deliberate playback, but never starts it.
    new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting) playback.pause();
      },
      { threshold: 0.3 },
    ).observe(container);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) playback.pause();
    });
  });
}

function readableOutcome(outcome) {
  const match = outcome.summary.match(
    /(\d+\/\d+)\s+(objects (?:newly )?in (?:the )?bin|objects in the bin at the end|in place)/i,
  );
  if (match) return `${match[1]} objects in the target at the end.`;
  if (outcome.summary.includes("Operator-scored"))
    return "Outcome scored on the real robot.";
  return outcome.success
    ? "Meets the recorded task success criterion."
    : "Does not meet the recorded task success criterion.";
}

function setupRobotics(data) {
  let kind = "replay";
  let currentId = "replay-01";
  let playback;
  const select = $("#episode-select");
  const buttons = $$("[data-robot-kind]");
  const itemsForKind = () => data.items.filter((item) => item.kind === kind);
  function renderEpisode() {
    const items = itemsForKind();
    const item = items.find((episode) => episode.id === currentId) || items[0];
    currentId = item.id;
    if (playback) playback.dispose();
    select.value = currentId;
    $("#episode-title").textContent = item.title;
    $("#episode-description").textContent = item.description;
    $("#robot-videos").innerHTML = ["real", "ours", "baseline"]
      .map(
        (role) =>
          `<figure class="robot-video-card ${role}">${videoHTML(item.media[role], `${data.methods[role]} — ${item.title}`)}<figcaption><span>${data.methods[role]}</span>${outcomeHTML(item.outcomes[role])}</figcaption><p>${readableOutcome(item.outcomes[role])}</p></figure>`,
      )
      .join("");
    $("#robot-play").textContent = "▶ Play all";
    playback = bindPlayback(
      $("#robot-videos"),
      $("#robot-play"),
      item.synchronized,
      data.playback_rates[item.kind],
    );
    const timing = item.synchronized
      ? `Synchronized replay at ${data.playback_rates.replay}× speed. Playing, pausing, or seeking one video controls all three; each receives the same recorded joint and gripper commands.`
      : `Independent policy rollouts at ${data.playback_rates.policy}× speed, with different durations. “Play all” starts them together for viewing; frames at the same time do not represent matched actions. Use each video’s controls to inspect its own trajectory.`;
    $("#robot-timing").textContent =
      `${timing}${item.presentation_note ? ` ${item.presentation_note}.` : ""}${item.note ? ` ${item.note}` : ""}`;
    $$(".episode-thumb").forEach((button) =>
      setActive(button, button.dataset.episode === currentId),
    );
    buttons.forEach((button) =>
      setActive(button, button.dataset.robotKind === kind),
    );
  }
  function renderEpisodes() {
    const items = itemsForKind();
    select.innerHTML = items
      .map(
        (item) =>
          `<option value="${item.id}">${escapeHTML(item.title)} · episode ${item.episode}</option>`,
      )
      .join("");
    $("#episode-thumbnails").innerHTML = items
      .map(
        (item) =>
          `<button class="episode-thumb" data-episode="${item.id}" aria-pressed="false" aria-label="View ${escapeHTML(item.title)}, episode ${item.episode}"><img src="${item.media.real.poster}" alt="" loading="lazy" width="192" height="108"><span>${escapeHTML(item.title)}<small>Episode ${item.episode} · ${item.outcomes.ours.success ? "SceneRig succeeds" : "SceneRig unsuccessful"}</small></span></button>`,
      )
      .join("");
    renderEpisode();
  }
  buttons.forEach((button) =>
    button.addEventListener("click", () => {
      kind = button.dataset.robotKind;
      currentId = kind === "replay" ? "replay-01" : "policy-01";
      renderEpisodes();
    }),
  );
  select.addEventListener("change", () => {
    currentId = select.value;
    renderEpisode();
  });
  $("#episode-thumbnails").addEventListener("click", (event) => {
    const button = event.target.closest("[data-episode]");
    if (!button) return;
    currentId = button.dataset.episode;
    renderEpisode();
  });
  $("#robot-restart").addEventListener("click", () => playback.restart());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) playback.pause();
  });
  new IntersectionObserver(
    (entries) => {
      if (!entries[0].isIntersecting && playback) playback.pause();
    },
    { threshold: 0.05 },
  ).observe($("#robot-videos"));
  renderEpisodes();
}

function setupPoseTrace(data) {
  const steps = $("#trace-steps");
  const detail = $("#trace-detail");
  let selected = 0;
  const number = (value) => Number(value).toFixed(3).replace(/0$/, "");

  $("#trace-context").textContent = data.summary;
  $("#trace-note").textContent = data.notes.join(" ");
  steps.innerHTML = data.calls
    .map(
      (call, index) =>
        `<button type="button" class="trace-step" data-trace-call="${call.id}" aria-pressed="false" aria-controls="trace-detail"><span class="trace-step-number">${index + 1}</span><span class="trace-step-label">${escapeHTML(call.label)}</span><code>${escapeHTML(call.tool)}</code></button>`,
    )
    .join("");

  function render(index) {
    selected = index;
    const call = data.calls[index];
    $$(".trace-step", steps).forEach((button, i) =>
      setActive(button, i === index),
    );
    const argumentsText = `${call.tool}(${JSON.stringify(call.arguments, null, 2)})`;
    detail.innerHTML = `
      <div class="trace-call-heading">
        <h4>${escapeHTML(call.label)} <small>· ${escapeHTML(data.target)}</small></h4>
        <div class="trace-navigation">
          <button type="button" class="icon-button" data-trace-direction="-1" aria-label="Previous tool call" ${index === 0 ? "disabled" : ""}>←</button>
          <span>${index + 1} / ${data.calls.length}</span>
          <button type="button" class="icon-button" data-trace-direction="1" aria-label="Next tool call" ${index === data.calls.length - 1 ? "disabled" : ""}>→</button>
        </div>
      </div>
      <div class="trace-body">
        <div class="trace-request">
          <p class="trace-role">Agent tool call</p>
          <pre class="trace-call-code"><code>${escapeHTML(argumentsText)}</code></pre>
          <p class="trace-role">Backend response (excerpt)</p>
          <pre class="trace-response">${escapeHTML(call.response)}</pre>
        </div>
        <div class="trace-evidence">
          <div class="trace-images">${call.images.map((item) => `<figure><img src="${item.src}" alt="${escapeHTML(item.alt)}" width="500" height="300" loading="lazy"><figcaption>${escapeHTML(item.label)}</figcaption></figure>`).join("")}</div>
          <dl class="trace-metrics">${call.metrics.map((metric) => `<div class="trace-metric"><dt>${escapeHTML(metric.label)}</dt><dd>${metric.value !== undefined ? number(metric.value) : `${number(metric.before)} <span aria-label="to">→</span> ${number(metric.after)}`}</dd></div>`).join("")}</dl>
          <p class="trace-summary">${escapeHTML(call.summary)}</p>
        </div>
      </div>`;
  }

  steps.addEventListener("click", (event) => {
    const button = event.target.closest("[data-trace-call]");
    if (button)
      render(
        data.calls.findIndex((call) => call.id === button.dataset.traceCall),
      );
  });
  steps.addEventListener("keydown", (event) => {
    const keys = {
      ArrowLeft: selected - 1,
      ArrowRight: selected + 1,
      Home: 0,
      End: data.calls.length - 1,
    };
    if (!(event.key in keys)) return;
    event.preventDefault();
    render(Math.max(0, Math.min(data.calls.length - 1, keys[event.key])));
    $$(".trace-step", steps)[selected].focus();
  });
  detail.addEventListener("click", (event) => {
    const button = event.target.closest("[data-trace-direction]");
    if (!button) return;
    const direction = Number(button.dataset.traceDirection);
    render(selected + direction);
    const nextButton = $(`[data-trace-direction="${direction}"]`, detail);
    (nextButton.disabled
      ? $(`[data-trace-direction="${-direction}"]`, detail)
      : nextButton
    ).focus();
  });
  render(0);
}

setupClaySlider();
readData("pose-trace")
  .then(setupPoseTrace)
  .catch(() => {
    $("#trace-detail").innerHTML =
      '<p class="error-message">The tool-call trace could not load. Refresh the page to try again.</p>';
  });
readData("gallery")
  .then(setupGallery)
  .catch(() => {
    $("#comparison-grid").innerHTML =
      '<p class="error-message">The gallery could not load. Refresh the page to try again.</p>';
  });
readData("robotics")
  .then((data) => {
    setupHeroVideos(data);
    setupRobotics(data);
  })
  .catch(() => {
    $("#hero-robotics").innerHTML =
      '<p class="error-message">The video gallery could not load. Refresh the page to try again.</p>';
  });

const correlationDialog = $("#policy-correlation");
$(".correlation-trigger").addEventListener("click", () => correlationDialog.showModal());
$(".correlation-close").addEventListener("click", () => correlationDialog.close());
correlationDialog.addEventListener("click", (event) => {
  if (event.target !== correlationDialog) return;
  const rect = correlationDialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right ||
      event.clientY < rect.top || event.clientY > rect.bottom) correlationDialog.close();
});

const reconstructionBaselines = {
  viga: { label: "VIGA", ari: "0.283", ospa: "4.9 cm", stability: "10.0%" },
  rest3d: { label: "REST3D", ari: "0.210", ospa: "4.7 cm", stability: "31.3%" },
  sceneconductor: { label: "SceneConductor", ari: "0.197", ospa: "5.0 cm", stability: "8.8%" },
  simfoundry: { label: "SimFoundry", ari: "0.676", ospa: "3.3 cm", stability: "98.8%" },
};
function updateReconstructionMetrics() {
  const baseline = reconstructionBaselines[$("#metric-baseline").value];
  $$("[data-baseline-metric]").forEach((element) => {
    element.textContent = `vs. ${baseline[element.dataset.baselineMetric]}`;
  });
  $("#metric-baseline-name").textContent = baseline.label;
}
$("#metric-baseline").addEventListener("change", updateReconstructionMetrics);
updateReconstructionMetrics();
