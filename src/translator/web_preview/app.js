/* FaceTempo W7: drive a procedural 3D head with 52 blendshape channels. */

const state = {
  data: null,
  duration: 0,
  playing: true,
  speed: 1,
  offset: 0,
  started: performance.now(),
  lastUI: 0,
};

const KEY_CHANNELS = [
  "eyeBlinkLeft", "eyeBlinkRight",
  "mouthSmileLeft", "mouthSmileRight",
  "jawOpen", "browInnerUp",
  "eyeSquintLeft", "eyeSquintRight",
];

const parts = {};
let renderer, scene, camera, head;
let channelRows = [];

init();

async function init() {
  try {
    const response = await fetch("/data/driving.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
  } catch (err) {
    showError(
      "无法读取 /data/driving.json<br>" +
      "请用 <code>python src/translator/serve_preview.py</code> 启动本地服务，" +
      "不要直接双击 HTML。<br><small>" + err.message + "</small>"
    );
    return;
  }

  const meta = state.data.meta;
  state.duration = meta.total_frames / meta.fps;

  document.getElementById("meta").innerHTML = `
    <div>source<br><b>${meta.source}</b></div>
    <div>frames<br><b>${meta.total_frames}</b></div>
    <div>fps<br><b>${meta.fps}</b></div>
    <div>channels<br><b>52</b></div>
  `;

  buildChannelList();
  setupScene();
  setupControls();
  animate();
}

function showError(html) {
  const el = document.getElementById("error");
  el.innerHTML = html;
  el.style.display = "grid";
}

function setupScene() {
  const canvas = document.getElementById("scene");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070d19);

  camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0.05, 4.4);
  camera.lookAt(0, -0.05, 0);

  scene.add(new THREE.HemisphereLight(0x9fd8ff, 0x0b1220, 0.42));

  const key = new THREE.DirectionalLight(0xffffff, 0.72);
  key.position.set(2.4, 3.2, 4.2);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x4d7cff, 0.22);
  fill.position.set(-3, 0.5, 2);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0x38bdf8, 0.55);
  rim.position.set(0, 1.5, -3.5);
  scene.add(rim);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(1.55, 0.012, 16, 120),
    new THREE.MeshBasicMaterial({ color: 0x1d4ed8, transparent: true, opacity: 0.65 })
  );
  ring.position.z = -1.2;
  scene.add(ring);

  head = new THREE.Group();
  head.position.y = -0.05;
  scene.add(head);

  const linear = (hex) => new THREE.Color(hex).convertSRGBToLinear();
  const skin = new THREE.MeshStandardMaterial({
    color: linear(0xe7b49b), roughness: 0.62, metalness: 0.04,
  });
  const skinDark = new THREE.MeshStandardMaterial({
    color: linear(0xd79c82), roughness: 0.7, metalness: 0.03,
  });
  const white = new THREE.MeshStandardMaterial({
    color: linear(0xf8fafc), roughness: 0.35,
  });
  const iris = new THREE.MeshStandardMaterial({
    color: linear(0x1e293b), roughness: 0.3,
  });
  const lip = new THREE.MeshStandardMaterial({
    color: linear(0x9e3b46), roughness: 0.55,
  });
  const mouthDark = new THREE.MeshStandardMaterial({
    color: linear(0x3b0d12), roughness: 0.8,
  });
  const browMat = new THREE.MeshStandardMaterial({
    color: linear(0x5b3a29), roughness: 0.8,
  });

  // 头骨
  const skull = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 48), skin);
  skull.scale.set(0.84, 1.02, 0.92);
  head.add(skull);

  // 耳朵
  for (const sx of [-1, 1]) {
    const ear = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 16), skinDark);
    ear.position.set(sx * 0.82, 0.0, -0.02);
    ear.scale.set(0.55, 1.0, 0.7);
    head.add(ear);
  }

  // 脖子
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.42, 0.75, 32), skinDark);
  neck.position.set(0, -1.08, -0.08);
  head.add(neck);

  // 眼睛 + 虹膜（眨眼通过缩放眼睛实现）
  parts.eyes = [];
  for (const sx of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.145, 32, 24), white);
    eye.position.set(sx * 0.32, 0.14, 0.76);
    eye.scale.set(1.05, 0.9, 0.55);
    head.add(eye);
    const irisMesh = new THREE.Mesh(new THREE.SphereGeometry(0.062, 24, 16), iris);
    irisMesh.position.set(sx * 0.32, 0.14, 0.84);
    irisMesh.scale.set(1, 1, 0.35);
    head.add(irisMesh);
    parts.eyes.push({ eye, iris: irisMesh, side: sx < 0 ? "Left" : "Right" });
  }

  // 眉毛
  for (const [key, sx] of [["browL", -1], ["browR", 1]]) {
    const brow = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.05, 0.05), browMat);
    brow.position.set(sx * 0.33, 0.5, 0.81);
    brow.rotation.z = sx * 0.12;
    head.add(brow);
    parts[key] = brow;
  }

  // 鼻子
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.095, 24, 18), skinDark);
  nose.position.set(0, -0.02, 0.9);
  nose.scale.set(0.62, 0.95, 0.55);
  head.add(nose);

  // 嘴
  parts.mouth = new THREE.Group();
  parts.mouth.position.set(0, -0.42, 0.86);
  head.add(parts.mouth);

  parts.inner = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.05, 0.02), mouthDark);
  parts.inner.position.set(0, -0.01, 0.0);
  parts.mouth.add(parts.inner);

  parts.upperLip = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.035, 0.05), lip);
  parts.upperLip.position.set(0, 0.03, 0.02);
  parts.mouth.add(parts.upperLip);

  parts.lowerLip = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.04, 0.05), lip);
  parts.lowerLip.position.set(0, -0.06, 0.02);
  parts.mouth.add(parts.lowerLip);

  for (const [key, sx] of [["cornerL", -1], ["cornerR", 1]]) {
    const corner = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), lip);
    corner.position.set(sx * 0.23, 0.0, 0.03);
    parts.mouth.add(corner);
    parts[key] = corner;
  }

  window.addEventListener("resize", resize);
  resize();
}

function resize() {
  const canvas = document.getElementById("scene");
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function buildChannelList() {
  const container = document.getElementById("channels");
  const names = Object.keys(state.data.frames[0].blendshapes);
  for (const name of names) {
    const row = document.createElement("div");
    row.className = "chan";
    row.innerHTML = `<span>${name}</span><span class="bar"><i></i></span><span class="val">0.000</span>`;
    container.appendChild(row);
    channelRows.push({ name, bar: row.querySelector("i"), val: row.querySelector(".val") });
  }
}

function setupControls() {
  const playBtn = document.getElementById("play");
  const scrub = document.getElementById("scrub");
  const speed = document.getElementById("speed");

  playBtn.addEventListener("click", togglePlay);
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  });

  scrub.addEventListener("input", () => {
    state.offset = parseFloat(scrub.value) * state.duration;
    state.started = performance.now();
  });

  speed.addEventListener("change", () => {
    state.offset = currentTime();
    state.started = performance.now();
    state.speed = parseFloat(speed.value);
  });

  function togglePlay() {
    if (state.playing) {
      state.offset = currentTime();
      playBtn.textContent = "▶ 播放";
    } else {
      state.started = performance.now();
      playBtn.textContent = "⏸ 暂停";
    }
    state.playing = !state.playing;
  }
}

function currentTime() {
  if (!state.playing) return state.offset % state.duration;
  const elapsed = (performance.now() - state.started) / 1000 * state.speed;
  return (state.offset + elapsed) % state.duration;
}

function sampleAt(t) {
  const { frames, meta } = state.data;
  const x = Math.min(t * meta.fps, frames.length - 1);
  const i = Math.floor(x);
  const j = Math.min(frames.length - 1, i + 1);
  const a = x - i;
  const A = frames[i].blendshapes;
  const B = frames[j].blendshapes;
  const out = {};
  for (const k in A) out[k] = A[k] + (B[k] - A[k]) * a;
  return { values: out, frame: i };
}

function applyValues(v) {
  const blinkL = v.eyeBlinkLeft;
  const blinkR = v.eyeBlinkRight;

  for (const { eye, iris, side } of parts.eyes) {
    const blink = side === "Left" ? blinkL : blinkR;
    const squint = side === "Left" ? v.eyeSquintLeft : v.eyeSquintRight;
    const open = Math.max(0.02, 1 - blink * 0.96 - squint * 0.10);
    eye.scale.y = 0.9 * open;
    iris.scale.y = Math.max(0.04, open);
  }

  const smileL = v.mouthSmileLeft;
  const smileR = v.mouthSmileRight;
  const smile = (smileL + smileR) / 2;
  parts.mouth.scale.x = 1 + smile * 0.35;
  parts.mouth.position.y = -0.42 + smile * 0.05;

  parts.cornerL.position.y = 0.0 + smileL * 0.13;
  parts.cornerR.position.y = 0.0 + smileR * 0.13;
  parts.cornerL.position.x = -0.23 - smileL * 0.05;
  parts.cornerR.position.x = 0.23 + smileR * 0.05;

  const jaw = v.jawOpen;
  parts.lowerLip.position.y = -0.05 - jaw * 0.28;
  parts.inner.scale.y = 1 + jaw * 7;

  const browUp = v.browInnerUp;
  parts.browL.position.y = 0.5 + browUp * 0.06 - v.browDownLeft * 0.09;
  parts.browR.position.y = 0.5 + browUp * 0.06 - v.browDownRight * 0.09;
}

function updateUI(values, frameIndex, t) {
  document.getElementById("frame").textContent = String(frameIndex);
  document.getElementById("time").textContent = t.toFixed(2);
  document.getElementById("scrub").value = (t / state.duration).toFixed(3);

  for (const row of channelRows) {
    const value = values[row.name] ?? 0;
    row.bar.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    row.val.textContent = value.toFixed(3);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const t = currentTime();
  const { values, frame } = sampleAt(t);
  applyValues(values);

  const now = performance.now();
  if (now - state.lastUI > 80) {
    updateUI(values, frame, t);
    state.lastUI = now;
  }

  head.rotation.y = Math.sin(t * 0.35) * 0.06;
  head.rotation.x = Math.sin(t * 0.27) * 0.02;

  renderer.render(scene, camera);
}
