/* FaceTempo W7/W8: drive a procedural 3D head with 52 blendshape channels. */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const state = {
  data: null,
  duration: 0,
  playing: true,
  speed: 1,
  offset: 0,
  started: performance.now(),
  lastUI: 0,
  usingGLB: false,
  activeRoot: null,
  morphBindings: [],
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
    const params = new URLSearchParams(location.search);
    const jsonName = params.get("json") || "driving.json";
    const response = await fetch(`/data/${encodeURIComponent(jsonName)}`, { cache: "no-store" });
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
    <div style="grid-column: 1 / -1;">renderer<br><b id="model-status">程序化示意头</b></div>
  `;

  buildChannelList();
  setupScene();
  loadOptionalGLB();
  setupControls();
  animate();
}

function showError(html) {
  const el = document.getElementById("error");
  el.innerHTML = html;
  el.style.display = "grid";
}

function buildEnvironment() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 256);
  gradient.addColorStop(0, "#5d6b7d");
  gradient.addColorStop(0.42, "#1b2029");
  gradient.addColorStop(1, "#05070a");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 512, 256);

  const glow = ctx.createRadialGradient(150, 70, 0, 150, 70, 150);
  glow.addColorStop(0, "rgba(255, 226, 196, 0.45)");
  glow.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, 512, 256);

  const texture = new THREE.CanvasTexture(canvas);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const env = pmrem.fromEquirectangular(texture).texture;
  pmrem.dispose();
  texture.dispose();
  return env;
}

function setupScene() {
  const canvas = document.getElementById("scene");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070d19);

  camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0.05, 4.4);
  camera.lookAt(0, -0.05, 0);

  scene.add(new THREE.HemisphereLight(0xcfd9e8, 0x1a120e, 0.5));

  const key = new THREE.DirectionalLight(0xfff4e6, 1.3);
  key.position.set(2.4, 3.2, 4.2);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = -2.5;
  key.shadow.camera.right = 2.5;
  key.shadow.camera.top = 2.8;
  key.shadow.camera.bottom = -2.8;
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 12;
  key.shadow.bias = -0.0004;
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x6f8dff, 0.16);
  fill.position.set(-3, 0.5, 2);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0x5cc8ff, 0.5);
  rim.position.set(0, 1.5, -3.5);
  scene.add(rim);

  scene.environment = buildEnvironment();

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(1.55, 0.012, 16, 120),
    new THREE.MeshBasicMaterial({ color: 0x1d4ed8, transparent: true, opacity: 0.35 })
  );
  ring.position.z = -1.2;
  scene.add(ring);

  head = new THREE.Group();
  head.position.y = -0.05;
  scene.add(head);
  state.activeRoot = head;
  state.basePos = head.position.clone();

  // 深色圆形台面，用来接住阴影
  const platform = new THREE.Mesh(
    new THREE.CircleGeometry(1.65, 64),
    new THREE.MeshStandardMaterial({ color: 0x0a1220, roughness: 0.92, metalness: 0.15 })
  );
  platform.rotation.x = -Math.PI / 2;
  platform.position.y = -1.22;
  platform.receiveShadow = true;
  scene.add(platform);

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

  head.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
  });

  window.__facetempo = { scene, state };
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

function normalizeName(name) {
  return name.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function channelCandidates(name) {
  const candidates = new Set([name]);
  if (name.endsWith("Left")) candidates.add(name.slice(0, -4) + "L");
  if (name.endsWith("Right")) candidates.add(name.slice(0, -5) + "R");
  candidates.add(name.replace(/_/g, ""));
  return [...candidates].map(normalizeName);
}

function loadOptionalGLB() {
  const box = document.getElementById("glb-error");
  if (box) {
    box.style.display = "block";
    box.textContent = "正在加载 model.glb…";
  }
  const ktx2 = new KTX2Loader()
    .setTranscoderPath("vendor/jsm/libs/basis/")
    .detectSupport(renderer);
  const loader = new GLTFLoader()
    .setKTX2Loader(ktx2)
    .setMeshoptDecoder(MeshoptDecoder);
  loader.load(
    "model.glb",
    (gltf) => {
      const model = gltf.scene;
      const wrapper = new THREE.Group();
      wrapper.add(model);

      const bounds = new THREE.Box3().setFromObject(model);
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      model.position.sub(center);
      const scale = 2.0 / Math.max(size.x, size.y, size.z || 1);
      wrapper.scale.setScalar(scale);

      const channels = Object.keys(state.data.frames[0].blendshapes);
      const bindings = [];
      model.traverse((obj) => {
        if (!obj.isMesh) return;
        obj.castShadow = true;
        obj.receiveShadow = true;
        const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const material of materials) {
          if (!material) continue;
          material.envMapIntensity = 1.1;
          material.roughness = Math.min(material.roughness ?? 1, 0.68);
          if (material.map) {
            material.map.colorSpace = THREE.SRGBColorSpace;
            material.color = new THREE.Color(0xffe2cc).convertSRGBToLinear();
          }
          material.needsUpdate = true;
        }
        if (!obj.morphTargetDictionary) return;
        const map = {};
        for (const [key, value] of Object.entries(obj.morphTargetDictionary)) {
          map[normalizeName(key)] = value;
        }
        for (const channel of channels) {
          for (const candidate of channelCandidates(channel)) {
            if (map[candidate] !== undefined) {
              bindings.push({ mesh: obj, index: map[candidate], channel });
              break;
            }
          }
        }
      });

      if (!bindings.length) {
        console.warn("model.glb 没有匹配的 morph targets，继续使用程序化头");
        const status = document.getElementById("model-status");
        if (status) status.textContent = "GLB 无匹配形态键";
        return;
      }

      head.visible = false;
      scene.add(wrapper);
      state.activeRoot = wrapper;
      state.basePos = wrapper.position.clone();
      state.usingGLB = true;
      state.morphBindings = bindings;
      const status = document.getElementById("model-status");
      if (status) status.textContent = `GLB · ${bindings.length} 通道`;
      if (box) box.style.display = "none";
      if (box) box.textContent = "";
    },
    undefined,
    (err) => {
      console.error("GLB load failed", err);
      const status = document.getElementById("model-status");
      if (status) status.textContent = "GLB 加载失败";
      const box = document.getElementById("glb-error");
      if (box) {
        box.style.display = "block";
        box.textContent = String((err && err.message) || err);
      }
    }
  );
}

function applyMorphs(values) {
  for (const binding of state.morphBindings) {
    const value = values[binding.channel] ?? 0;
    binding.mesh.morphTargetInfluences[binding.index] = value;
  }
}

function animate() {
  requestAnimationFrame(animate);
  const t = currentTime();
  const { values, frame } = sampleAt(t);
  applyValues(values);
  if (state.usingGLB) applyMorphs(values);

  const now = performance.now();
  if (now - state.lastUI > 80) {
    updateUI(values, frame, t);
    state.lastUI = now;
  }

  const idle = performance.now() / 1000;
  const root = state.activeRoot || head;
  const base = state.basePos || { y: 0 };
  root.position.y = base.y + Math.sin(idle * 0.9) * 0.008;
  root.rotation.y = Math.sin(idle * 0.45) * 0.05;
  root.rotation.x = Math.sin(idle * 0.7) * 0.012;

  renderer.render(scene, camera);
}
