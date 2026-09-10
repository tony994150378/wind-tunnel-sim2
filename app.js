// ============================================================
//  Wind Tunnel CFD Simulator — Lattice Boltzmann Method (D3Q19)
// ============================================================

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---- Compatibility check ----
(function checkCompat() {
  var canvas = document.createElement('canvas');
  var gl = canvas.getContext('webgl2');
  if (!gl) {
    var el = document.getElementById('compat-warn');
    if (el) {
      el.style.display = 'flex';
      el.querySelector('h2').textContent = '⚠️ 浏览器不兼容';
      el.querySelector('p').textContent = '您的浏览器不支持 WebGL2，无法运行风洞模拟器。请升级浏览器。';
    }
    var vp = document.getElementById('viewport');
    if (vp) vp.style.display = 'none';
    var tb = document.getElementById('toolbar');
    if (tb) tb.style.display = 'none';
    throw new Error('WebGL2 not supported');
  }
})();

// ---- Color palettes ----
var RAINBOW = [
  [0,0,1],[0,.5,1],[0,1,1],[0,1,.5],[0,1,0],
  [.5,1,0],[1,1,0],[1,.5,0],[1,0,0]
];
var COOLWARM = [
  [0,.2,1],[.2,.5,1],[.5,.8,1],[.8,.9,.9],[.95,.95,.95],
  [.9,.8,.7],[1,.5,.3],[1,.2,.1],[.8,0,0]
];
var CONTOUR = [
  [0,0,.4],[0,.2,.8],[0,.6,1],[0,1,.8],[.5,1,.3],
  [1,1,0],[1,.6,0],[1,.2,0],[.6,0,0]
];

// ---- GPU Performance Profiler ----
function profileGPU() {
  try {
  var canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  var gl = canvas.getContext('webgl2');
  if (!gl) return { score: 20, gpu: 'unknown' };

  var debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  var gpuRenderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown';
  var gpuVendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'unknown';
  var maxTexSize = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
  var maxFragUniforms = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS);
  var isMesa = gpuRenderer.toLowerCase().indexOf('mesa') >= 0 || gpuRenderer.toLowerCase().indexOf('llvmpipe') >= 0;
  var isSwiftShader = gpuRenderer.toLowerCase().indexOf('swiftshader') >= 0;
  var isIntel = gpuRenderer.toLowerCase().indexOf('intel') >= 0;
  var cores = navigator.hardwareConcurrency || 4;
  var memory = navigator.deviceMemory || 4;

  // Quick GPU benchmark: draw 800K points and measure
  var vs = 'attribute vec3 aPos; attribute float aVal; varying float vVal; void main(){ vVal=aVal; gl_Position=vec4(aPos,1); gl_PointSize=1.0; }';
  var fs = 'precision mediump float; varying float vVal; void main(){ gl_FragColor=vec4(vVal,vVal,vVal,1); }';
  var prog = gl.createProgram();
  function mkShader(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.warn('[GPU Profile] Shader compile failed:', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    gl.attachShader(prog, s);
    return s;
  }
  var vsh = mkShader(gl.VERTEX_SHADER, vs);
  var fsh = mkShader(gl.FRAGMENT_SHADER, fs);
  if (!vsh || !fsh) {
    // Cleanup and return fallback
    if (vsh) gl.deleteShader(vsh);
    if (fsh) gl.deleteShader(fsh);
    gl.deleteProgram(prog);
    return { score: 30, gpu: gpuRenderer, vendor: gpuVendor, cores: cores, memory: memory, maxTexSize: maxTexSize };
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.warn('[GPU Profile] Program link failed:', gl.getProgramInfoLog(prog));
    gl.deleteShader(vsh); gl.deleteShader(fsh); gl.deleteProgram(prog);
    return { score: 30, gpu: gpuRenderer, vendor: gpuVendor, cores: cores, memory: memory, maxTexSize: maxTexSize };
  }
  gl.useProgram(prog);

  var N = 800000;
  var posData = new Float32Array(N * 3);
  var valData = new Float32Array(N);
  for (var i = 0; i < N; i++) {
    posData[i*3] = Math.random()*2-1;
    posData[i*3+1] = Math.random()*2-1;
    posData[i*3+2] = 0;
    valData[i] = Math.random();
  }
  var posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, posData, gl.STATIC_DRAW);
  var aPos = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

  var valBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, valBuf);
  gl.bufferData(gl.ARRAY_BUFFER, valData, gl.STATIC_DRAW);
  var aVal = gl.getAttribLocation(prog, 'aVal');
  gl.enableVertexAttribArray(aVal);
  gl.vertexAttribPointer(aVal, 1, gl.FLOAT, false, 0, 0);

  gl.viewport(0, 0, 128, 128);
  gl.clearColor(0,0,0,1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Warmup
  gl.drawArrays(gl.POINTS, 0, Math.min(N, 100000));
  gl.finish();

  // Benchmark
  var t0 = performance.now();
  gl.drawArrays(gl.POINTS, 0, N);
  gl.finish();
  var elapsed = performance.now() - t0;

  // Cleanup
  gl.deleteBuffer(posBuf);
  gl.deleteBuffer(valBuf);
  gl.deleteProgram(prog);
  gl.deleteShader(vsh);
  gl.deleteShader(fsh);

  // Score: lower elapsed = higher score (0-100)
  var score;
  if (elapsed < 4) score = 95;
  else if (elapsed < 8) score = 80;
  else if (elapsed < 15) score = 65;
  else if (elapsed < 25) score = 50;
  else if (elapsed < 40) score = 35;
  else score = 20;

  // Penalty for software rendering / weak GPUs
  if (isSwiftShader) score = Math.min(score, 15);
  if (isMesa) score = Math.min(score, 30);
  if (isIntel && elapsed > 20) score = Math.min(score, 35);
  if (maxTexSize < 256) score = Math.min(score, 25);

  return {
    score: score,
    elapsed: elapsed,
    gpu: gpuRenderer,
    vendor: gpuVendor,
    cores: cores,
    memory: memory,
    maxTexSize: maxTexSize
  };
  } catch (e) {
    console.warn('[GPU Profile] Error:', e.message);
    return { score: 30, gpu: 'unknown', vendor: 'unknown', cores: navigator.hardwareConcurrency || 4, memory: navigator.deviceMemory || 4, maxTexSize: 256 };
  }
}

// ---- Quality presets ----
var QUALITY_PRESETS = {
  low:    { grid: 48, particles: 60000,  stepsPerFrame: 2, label: '流畅模式 · 低配优化' },
  medium: { grid: 64, particles: 100000, stepsPerFrame: 3, label: '均衡模式 · 推荐' },
  high:   { grid: 80, particles: 140000, stepsPerFrame: 3, label: '高清模式 · 细节丰富' },
  ultra:  { grid: 96, particles: 180000, stepsPerFrame: 4, label: '极致画质' }
};

function pickPreset(score) {
  if (score >= 75) return 'ultra';
  if (score >= 55) return 'high';
  if (score >= 30) return 'medium';
  return 'low';
}

// ============================================================
//  Main Application
// ============================================================
function WindTunnelApp() {
  this.SIM_SIZE = 48;
  this.running = false;
  this.stepCount = 0;
  this.frameCount = 0;
  this.lastFpsTime = performance.now();
  this.currentModel = 'airfoil';
  this.modelScale = 1.0;
  this.velocity = 0.08;
  this.viscosity = 0.02;
  this.particleCount = 100000;
  this.stepsPerFrame = 3;
  this.qualityPreset = 'medium';
  this.gpuProfile = null;
  this.streamlineDensity = 5;

  this.initThree();
  this.addLights();
  this.addGround();
  this.addTunnel();
  this.initVelocityField();
  this.initParticles(this.particleCount);
  this.initSlice();
  this.initWorker();
  this.initUI();
  this.loadModel('airfoil');
  this.updateColorBar();
  this.autoProfile();
  this.animate();
}

// ---- Three.js Setup ----
WindTunnelApp.prototype.initThree = function () {
  this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  this.renderer.setSize(window.innerWidth, window.innerHeight);
  this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  if (this.renderer.outputColorSpace !== undefined) this.renderer.outputColorSpace = THREE.SRGBColorSpace;
  if (this.renderer.toneMapping !== undefined) { this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.0; }
  document.getElementById('viewport').appendChild(this.renderer.domElement);

  this.scene = new THREE.Scene();
  this.scene.background = new THREE.Color(0x0a1220);
  this.scene.fog = new THREE.FogExp2(0x0a1220, 0.0015);

  this.camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 500);
  this.camera.position.set(70, 45, 70);
  this.camera.lookAt(0, 0, 0);

  this.controls = new OrbitControls(this.camera, this.renderer.domElement);
  this.controls.enableDamping = true;
  this.controls.dampingFactor = 0.08;
  this.controls.maxDistance = 250;
  this.controls.target.set(0, 0, 0);
  this.controls.update();

  var self = this;
  window.addEventListener('resize', function () {
    self.camera.aspect = window.innerWidth / window.innerHeight;
    self.camera.updateProjectionMatrix();
    self.renderer.setSize(window.innerWidth, window.innerHeight);
  });
};

WindTunnelApp.prototype.addLights = function () {
  this.scene.add(new THREE.AmbientLight(0x4488cc, 0.8));
  var d1 = new THREE.DirectionalLight(0xffffff, 1.0);
  d1.position.set(50, 80, 60);
  d1.castShadow = false;
  this.scene.add(d1);
  var d2 = new THREE.DirectionalLight(0x4488cc, 0.5);
  d2.position.set(-40, 20, -30);
  this.scene.add(d2);
  // Add rim light for better model definition
  var d3 = new THREE.DirectionalLight(0x88aaff, 0.3);
  d3.position.set(0, -30, 50);
  this.scene.add(d3);
  // Hemisphere light for natural ambient
  this.scene.add(new THREE.HemisphereLight(0x88aacc, 0x223344, 0.4));
};

WindTunnelApp.prototype.addGround = function () {
  var S = this.SIM_SIZE;
  var ground = new THREE.Mesh(
    new THREE.PlaneGeometry(300, 300),
    new THREE.MeshStandardMaterial({ color: 0x0a1520, roughness: 0.9, metalness: 0.1 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -S * 0.52;
  this.scene.add(ground);
  var grid = new THREE.GridHelper(200, 40, 0x1a3050, 0x152535);
  grid.position.y = -S * 0.515;
  this.scene.add(grid);
};

WindTunnelApp.prototype.addTunnel = function () {
  var S = this.SIM_SIZE;
  var hw = S * 0.52;
  // Wireframe box
  var tunnel = new THREE.Mesh(
    new THREE.BoxGeometry(S * 1.1, S * 1.05, S * 1.05),
    new THREE.MeshBasicMaterial({ color: 0x4fc3f7, wireframe: true, transparent: true, opacity: 0.04 })
  );
  this.scene.add(tunnel);
  var edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(S * 1.1, S * 1.05, S * 1.05));
  this.scene.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.12 })));

  // Wind arrows
  var coneGeo = new THREE.ConeGeometry(1.5, 4, 8);
  var coneMat = new THREE.MeshBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.3 });
  var offsets = [[0,10,0],[0,-10,0],[0,0,15],[0,0,-15]];
  for (var i = 0; i < offsets.length; i++) {
    var c = new THREE.Mesh(coneGeo, coneMat);
    c.position.set(-hw - 8, offsets[i][1], offsets[i][2]);
    c.rotation.z = -Math.PI / 2;
    this.scene.add(c);
  }

  // Reference lines
  var lineMat = new THREE.LineBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.15 });
  var pairs = [[hw,hw],[hw,-hw],[-hw,hw],[-hw,-hw]];
  for (var j = 0; j < pairs.length; j++) {
    var g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-hw - 15, pairs[j][0], pairs[j][1]),
      new THREE.Vector3(hw + 15, pairs[j][0], pairs[j][1])
    ]);
    this.scene.add(new THREE.Line(g, lineMat));
  }

  // Reference spheres
  var refMat = new THREE.MeshBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.2 });
  var refGeo = new THREE.SphereGeometry(1.2, 8, 8);
  var signs = [-1, 1];
  for (var a = 0; a < 2; a++)
    for (var b = 0; b < 2; b++)
      for (var cc = 0; cc < 2; cc++) {
        var m = new THREE.Mesh(refGeo, refMat);
        m.position.set(signs[a] * (hw + 15), signs[b] * hw, signs[cc] * hw);
        this.scene.add(m);
      }
};

// ---- Velocity field 3D texture ----
WindTunnelApp.prototype.initVelocityField = function () {
  var S = this.SIM_SIZE;
  var texData = new Float32Array(S * S * S * 4);
  this.velTexture = new THREE.Data3DTexture(texData, S, S, S);
  this.velTexture.format = THREE.RGBAFormat;
  this.velTexture.type = THREE.FloatType;
  this.velTexture.minFilter = THREE.LinearFilter;
  this.velTexture.magFilter = THREE.LinearFilter;
  this.velTexture.needsUpdate = true;
};

WindTunnelApp.prototype.makeParticleTexture = function () {
  var size = 64;
  var canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  var ctx = canvas.getContext('2d');
  var half = size / 2;
  var gradient = ctx.createRadialGradient(half, half, 0, half, half, half);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.2, 'rgba(255,255,255,0.8)');
  gradient.addColorStop(0.5, 'rgba(255,255,255,0.3)');
  gradient.addColorStop(0.8, 'rgba(255,255,255,0.05)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  var tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
};

// ---- Color bar ----
WindTunnelApp.prototype.updateColorBar = function () {
  var canvas = document.getElementById('colorbar-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var w = canvas.width, h = canvas.height;
  var cmap = parseInt(document.getElementById('ctrl-colormap').value);
  var palettes = [RAINBOW, COOLWARM, CONTOUR];
  var pal = palettes[cmap] || RAINBOW;

  for (var x = 0; x < w; x++) {
    var t = x / (w - 1);
    var idx = t * (pal.length - 1);
    var lo = Math.floor(idx), hi = Math.min(lo + 1, pal.length - 1);
    var f = idx - lo;
    var r = Math.round((pal[lo][0] + (pal[hi][0] - pal[lo][0]) * f) * 255);
    var g = Math.round((pal[lo][1] + (pal[hi][1] - pal[lo][1]) * f) * 255);
    var b = Math.round((pal[lo][2] + (pal[hi][2] - pal[lo][2]) * f) * 255);
    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    ctx.fillRect(x, 0, 1, h);
  }

  // Update labels with actual velocity values
  var maxV = this.velocity * 1.5;
  var setLabel = function(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val.toFixed(2);
  };
  setLabel('colorbar-max', maxV);
  setLabel('colorbar-q3', maxV * 0.75);
  setLabel('colorbar-q2', maxV * 0.5);
  setLabel('colorbar-q1', maxV * 0.25);
};

// ---- Particle system ----
WindTunnelApp.prototype.initParticles = function (count) {
  if (this.particleSystem) {
    this.scene.remove(this.particleSystem);
    this.particleSystem.geometry.dispose();
    this.particleSystem.material.dispose();
  }
  this.particleCount = count;
  var S = this.SIM_SIZE;
  this.particlePos = new Float32Array(count * 3);
  this.particleColors = new Float32Array(count * 3);
  this.particleAges = new Float32Array(count);
  this.particleRandoms = new Float32Array(count);

  for (var i = 0; i < count; i++) {
    this.particlePos[i*3]   = (Math.random() - 0.5) * S;
    this.particlePos[i*3+1] = (Math.random() - 0.5) * S * 0.9;
    this.particlePos[i*3+2] = (Math.random() - 0.5) * S * 0.9;
    this.particleAges[i] = Math.random() * 3.0;
    this.particleRandoms[i] = Math.random();
    this.particleColors[i*3] = 0;
    this.particleColors[i*3+1] = 0.5;
    this.particleColors[i*3+2] = 1;
  }

  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(this.particlePos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(this.particleColors, 3));

  var mat = new THREE.PointsMaterial({
    size: 1.2,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true
  });

  this.particleSystem = new THREE.Points(geo, mat);
  this.scene.add(this.particleSystem);
  this.velFieldData = null;
  console.log('[App] Particles created:', count);
};

WindTunnelApp.prototype.updateParticles = function () {
  if (!this.velFieldData || !this.particlePos) return;
  var S = this.SIM_SIZE;
  var N = S * S * S;
  var vel = this.velFieldData;
  var pos = this.particlePos;
  var col = this.particleColors;
  var ages = this.particleAge || this.particleAges;
  var rnds = this.particleRandoms;
  var halfS = S / 2;
  var speedMul = 2.0;
  var count = this.particleCount;

  for (var i = 0; i < count; i++) {
    var i3 = i * 3;
    var px = pos[i3], py = pos[i3+1], pz = pos[i3+2];
    var age = ages[i];

    // Advect through velocity field (3 steps)
    for (var step = 0; step < 3; step++) {
      var tx = (px + halfS) / S;
      var ty = (py + halfS) / S;
      var tz = (pz + halfS) / S;
      if (tx < 0 || tx >= 1 || ty < 0 || ty >= 1 || tz < 0 || tz >= 1) break;
      var ix = Math.floor(tx * (S-1));
      var iy = Math.floor(ty * (S-1));
      var iz = Math.floor(tz * (S-1));
      var idx = (ix + iy * S + iz * S * S) * 3;
      if (idx >= 0 && idx + 2 < vel.length) {
        px += vel[idx] * speedMul;
        py += vel[idx+1] * speedMul;
        pz += vel[idx+2] * speedMul;
      }
      age += 0.02;
    }

    // Respawn if out of bounds or too old
    if (age > 3.0 || px > halfS*1.1 || px < -halfS*1.1 ||
        Math.abs(py) > halfS*0.95 || Math.abs(pz) > halfS*0.95) {
      px = -halfS * 1.1 - rnds[i] * halfS * 0.2;
      py = (rnds[i] * 2 - 1) * halfS * 0.85;
      pz = ((rnds[i] * 7) % 1 * 2 - 1) * halfS * 0.85;
      age = 0;
    }

    // Color by velocity
    var tx2 = (px + halfS) / S;
    var ty2 = (py + halfS) / S;
    var tz2 = (pz + halfS) / S;
    var spd = 0;
    if (tx2 >= 0 && tx2 < 1 && ty2 >= 0 && ty2 < 1 && tz2 >= 0 && tz2 < 1) {
      var ix2 = Math.floor(tx2 * (S-1));
      var iy2 = Math.floor(ty2 * (S-1));
      var iz2 = Math.floor(tz2 * (S-1));
      var idx2 = (ix2 + iy2 * S + iz2 * S * S) * 3;
      if (idx2 >= 0 && idx2 + 2 < vel.length) {
        spd = Math.sqrt(vel[idx2]*vel[idx2]+vel[idx2+1]*vel[idx2+1]+vel[idx2+2]*vel[idx2+2]);
      }
    }
    var t = Math.min(spd * 5, 1);
    var r, g, b;
    if (t < 0.25) { r = 0; g = t*4; b = 1; }
    else if (t < 0.5) { r = 0; g = 1; b = 1-(t-0.25)*4; }
    else if (t < 0.75) { r = (t-0.5)*4; g = 1; b = 0; }
    else { r = 1; g = 1-(t-0.75)*4; b = 0; }

    pos[i3] = px; pos[i3+1] = py; pos[i3+2] = pz;
    ages[i] = age;
    col[i3] = r; col[i3+1] = g; col[i3+2] = b;
  }

  this.particleSystem.geometry.attributes.position.needsUpdate = true;
  this.particleSystem.geometry.attributes.color.needsUpdate = true;
};

// ---- Slice visualization ----
WindTunnelApp.prototype.initSlice = function () {
  var S = this.SIM_SIZE;
  var geo = new THREE.PlaneGeometry(S * 0.95, S * 0.95);

  var vertShader = [
    'varying vec2 vUv;',
    'void main() {',
    '  vUv = uv;',
    '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
    '}'
  ].join('\n');

  var fragShader = [
    'uniform sampler3D uVelocityField;',
    'uniform float uSlicePos;',
    'uniform int uColormap;',
    'uniform int uSliceAxis;',
    'varying vec2 vUv;',
    'vec3 getColor(float t, int cmap) {',
    '  if (cmap == 1) {',
    '    if (t<0.25) return mix(vec3(0,.2,1),vec3(.5,.8,1),t*4.0);',
    '    if (t<0.5)  return mix(vec3(.5,.8,1),vec3(.95),(t-0.25)*4.0);',
    '    if (t<0.75) return mix(vec3(.95),vec3(1,.5,.3),(t-0.5)*4.0);',
    '    return mix(vec3(1,.5,.3),vec3(.8,0,0),(t-0.75)*4.0);',
    '  } else if (cmap == 2) {',
    '    float s = sin(t*18.0)*0.5+0.5;',
    '    return mix(vec3(0,.3,.8),vec3(.8,.1,0),s);',
    '  } else {',
    '    if (t<0.25) return mix(vec3(0,0,1),vec3(0,1,1),t*4.0);',
    '    if (t<0.5)  return mix(vec3(0,1,1),vec3(0,1,0),(t-0.25)*4.0);',
    '    if (t<0.75) return mix(vec3(0,1,0),vec3(1,1,0),(t-0.5)*4.0);',
    '    return mix(vec3(1,1,0),vec3(1,0,0),(t-0.75)*4.0);',
    '  }',
    '}',
    'void main() {',
    '  vec3 tc;',
    '  if (uSliceAxis == 1) { tc = vec3(vUv.x, uSlicePos, vUv.y); }',
    '  else if (uSliceAxis == 2) { tc = vec3(vUv.x, vUv.y, uSlicePos); }',
    '  else { tc = vec3(uSlicePos, vUv.x, vUv.y); }',
    '  vec3 vel = texture(uVelocityField, clamp(tc, 0.005, 0.995)).rgb;',
    '  float speed = length(vel);',
    '  float t = clamp(speed * 6.0, 0.0, 1.0);',
    '  vec3 color = getColor(t, uColormap);',
    '  gl_FragColor = vec4(color, 0.85);',
    '}'
  ].join('\n');

  var mat = new THREE.ShaderMaterial({
    vertexShader: vertShader,
    fragmentShader: fragShader,
    uniforms: {
      uVelocityField: { value: this.velTexture },
      uSlicePos: { value: 0.5 },
      uColormap: { value: 0 },
      uSliceAxis: { value: 0 }
    },
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false
  });

  this.sliceMesh = new THREE.Mesh(geo, mat);
  this.sliceMesh.rotation.y = Math.PI / 2;
  this.sliceMesh.position.x = 0;
  this.sliceAxis = 0; // 0=X, 1=Y, 2=Z
  this.sliceMesh.visible = false;
  this.scene.add(this.sliceMesh);
};

// ---- Worker (LBM engine) ----
WindTunnelApp.prototype.initWorker = function () {
  var workerCode = [
    '// LBM D3Q19 Worker',
    'var Q=19;',
    'var cx=[0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0];',
    'var cy=[0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1];',
    'var cz=[0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1];',
    'var w=[1/3,1/18,1/18,1/18,1/18,1/18,1/18,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36];',
    'var opp=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17];',
    'var SX,SY,SZ,N,fIn,fOut,rho,ux,uy,uz,obstacle;',
    'var omega=1.0,uMax=0.08;',
    '',
    'function doStep() {',
    '  var x,y,z,i,idx,r,rInv,vx,vy,vz,f,u2,eu,nidx;',
    '  var SXY=SX*SY,SXm1=SX-1,SYm1=SY-1,SZm1=SZ-1;',
    '  // Collision',
    '  for (z=0;z<SZ;z++) for (y=0;y<SY;y++) for (x=0;x<SX;x++) {',
    '    idx=x+y*SX+z*SXY;',
    '    if (obstacle[idx]) continue;',
    '    r=0;vx=0;vy=0;vz=0;',
    '    for (i=0;i<19;i++) { f=fIn[i*N+idx]; r+=f; vx+=cx[i]*f; vy+=cy[i]*f; vz+=cz[i]*f; }',
    '    if (r<1e-12) r=1e-12;',
    '    rInv=1/r; vx*=rInv; vy*=rInv; vz*=rInv;',
    '    rho[idx]=r; ux[idx]=vx; uy[idx]=vy; uz[idx]=vz;',
    '    u2=vx*vx+vy*vy+vz*vz;',
    '    var om1=1-omega,omW=omega*r,term3=1-1.5*u2;',
    '    for (i=0;i<19;i++) {',
    '      eu=cx[i]*vx+cy[i]*vy+cz[i]*vz;',
    '      fOut[i*N+idx]=om1*fIn[i*N+idx]+omW*w[i]*(term3+3*eu+4.5*eu*eu);',
    '    }',
    '  }',
    '  // Streaming + bounce-back',
    '  for (z=0;z<SZ;z++) for (y=0;y<SY;y++) for (x=0;x<SX;x++) {',
    '    idx=x+y*SX+z*SXY;',
    '    for (i=0;i<19;i++) {',
    '      var nx=x-cx[i],ny=y-cy[i],nz=z-cz[i];',
    '      if (nx<0||nx>SXm1||ny<0||ny>SYm1||nz<0||nz>SZm1) continue;',
    '      nidx=nx+ny*SX+nz*SXY;',
    '      fIn[i*N+idx]=obstacle[nidx]?fOut[opp[i]*N+idx]:fOut[i*N+nidx];',
    '    }',
    '  }',
    '  // Inlet BC (x=0)',
    '  for (z=0;z<SZ;z++) for (y=0;y<SY;y++) {',
    '    idx=y*SX+z*SXY;',
    '    if (obstacle[idx]) continue;',
    '    r=0; for (i=0;i<19;i++) r+=fIn[i*N+idx];',
    '    var rv=r/(1-uMax);',
    '    fIn[1*N+idx]=fIn[2*N+idx]+(2/3)*rv*uMax;',
    '    fIn[7*N+idx]=fIn[8*N+idx]+0.5*(fIn[4*N+idx]-fIn[3*N+idx])+(1/6)*rv*uMax;',
    '    fIn[9*N+idx]=fIn[10*N+idx]+0.5*(fIn[3*N+idx]-fIn[4*N+idx])+(1/6)*rv*uMax;',
    '    fIn[11*N+idx]=fIn[12*N+idx]+0.5*(fIn[6*N+idx]-fIn[5*N+idx])+(1/6)*rv*uMax;',
    '    fIn[13*N+idx]=fIn[14*N+idx]+0.5*(fIn[5*N+idx]-fIn[6*N+idx])+(1/6)*rv*uMax;',
    '  }',
    '  // Outlet BC (x=SX-1)',
    '  for (z=0;z<SZ;z++) for (y=0;y<SY;y++) {',
    '    x=SX-1; idx=x+y*SX+z*SXY; nidx=(x-1)+y*SX+z*SXY;',
    '    for (i=0;i<19;i++) fIn[i*N+idx]=fIn[i*N+nidx];',
    '  }',
    '}',
    '',
    'function getVelocityData() {',
    '  var buf=new Float32Array(N*3);',
    '  for (var i=0;i<N;i++) { buf[i*3]=ux[i]; buf[i*3+1]=uy[i]; buf[i*3+2]=uz[i]; }',
    '  return buf;',
    '}',
    '',
    'var dataRequested=false,stepTimer=null,stepsPerFrame=3;',
    'self.onmessage=function(e){',
    '  var d=e.data;',
    '  if (d.type==="init") {',
    '    SX=d.sizeX;SY=d.sizeY;SZ=d.sizeZ;N=SX*SY*SZ;',
    '    uMax=d.velocity||0.08;',
    '    omega=1/(3*(d.viscosity||0.02)+0.5);',
    '    stepsPerFrame=d.stepsPerFrame||3;',
    '    rho=new Float32Array(N);ux=new Float32Array(N);uy=new Float32Array(N);uz=new Float32Array(N);',
    '    if(!obstacle) obstacle=new Uint8Array(N);',
    '    fIn=new Float32Array(N*Q);fOut=new Float32Array(N*Q);',
    '    for (var i=0;i<N;i++) {',
    '      rho[i]=1;ux[i]=0;uy[i]=0;uz[i]=0;',
    '      for (var q=0;q<Q;q++) fIn[q*N+i]=w[q];',
    '      if (!obstacle[i]) {',
    '        for (var q=0;q<Q;q++) {',
    '          var eu=cx[q]*uMax;',
    '          fIn[q*N+i]=w[q]*(1+3*eu+4.5*eu*eu-1.5*uMax*uMax);',
    '        }',
    '      }',
    '    }',
    '    if (stepTimer) clearInterval(stepTimer);',
    '    stepTimer=setInterval(function(){',
    '      for (var s=0;s<stepsPerFrame;s++) doStep();',
    '      self.postMessage({type:"stepDone"});',
    '      if (dataRequested) {',
    '        var velBuf=getVelocityData();',
    '        self.postMessage({type:"data",buffer:velBuf.buffer},[velBuf.buffer]);',
    '        dataRequested=false;',
    '      }',
    '    },30);',
    '    self.postMessage({type:"ready"});',
    '  }',
    '  else if (d.type==="loadObstacle") {',
    '    obstacle=new Uint8Array(d.data);',
    '    self.postMessage({type:"obstacleLoaded"});',
    '  }',
    '  else if (d.type==="setVelocity") {',
    '    uMax=d.value; omega=1/(3*(d.viscosity||0.02)+0.5);',
    '  }',
    '  else if (d.type==="setStepsPerFrame") { stepsPerFrame=d.value||3; }',
    '  else if (d.type==="requestData") { dataRequested=true; }',
    '  else if (d.type==="terminate") { if(stepTimer)clearInterval(stepTimer); self.close(); }',
    '};'
  ].join('\n');

  var blob = new Blob([workerCode], { type: 'application/javascript' });
  this.worker = new Worker(URL.createObjectURL(blob));
  this.velocityData = null;

  var self = this;
  this.worker.onmessage = function (e) {
    var d = e.data;
    console.log('[Worker→App]', d.type);
    if (d.type === 'ready') self.onWorkerReady();
    else if (d.type === 'obstacleLoaded') self.onObstacleLoaded();
    else if (d.type === 'stepDone') self.onStepDone();
    else if (d.type === 'data') self.onVelocityData(d.buffer);
  };
  this.worker.onerror = function (e) {
    console.error('[Worker error]', e.message);
    document.getElementById('status-text').textContent = 'Worker错误: ' + e.message;
  };
};

WindTunnelApp.prototype.onWorkerReady = function () {
  console.log('[App] Worker ready');
  document.getElementById('status-text').textContent = '模拟引擎就绪';
};

WindTunnelApp.prototype.onObstacleLoaded = function () {
  console.log('[App] Obstacle loaded, starting worker...');
  document.getElementById('status-text').textContent = '模型已加载 - 正在初始化模拟引擎...';
  this.startWorker();
};

WindTunnelApp.prototype.onStepDone = function () {
  this.stepCount++;
  if (this.running) this.worker.postMessage({ type: 'requestData' });
};

WindTunnelApp.prototype.onVelocityData = function (buffer) {
  this.velocityData = new Float32Array(buffer);
  this.velFieldData = this.velocityData;
  var texData = this.velTexture.image.data;
  var N = this.SIM_SIZE * this.SIM_SIZE * this.SIM_SIZE;
  for (var i = 0; i < N; i++) {
    texData[i*4]   = this.velocityData[i*3];
    texData[i*4+1] = this.velocityData[i*3+1];
    texData[i*4+2] = this.velocityData[i*3+2];
    texData[i*4+3] = 0;
  }
  this.velTexture.needsUpdate = true;
  this.updateParticles();
  if (this.stepCount % 20 === 0) this.updateStreamlines();
};

// ---- Model loading ----
WindTunnelApp.prototype.loadModel = function (name) {
  console.log('[App] Loading model:', name);
  var mesh;
  if (name === 'sphere') mesh = this.createSphere();
  else if (name === 'car') mesh = this.createCar();
  else if (name === 'airfoil') mesh = this.createAirfoil();
  else if (name === 'cube') mesh = this.createCube();
  else if (name === 'cylinder') mesh = this.createCylinder();
  else if (name === 'f1') mesh = this.createF1Car();
  else if (name === 'bridge') mesh = this.createBridge();
  else if (name === 'airplane') mesh = this.createAirplane();
  else mesh = this.createSphere();
  this.currentModelMesh = mesh;
  this.currentModel = name;
  this.showModelMesh(mesh);
  this.voxelizeAndLoad(mesh);
};

// ---- Streamlines ----
WindTunnelApp.prototype.updateStreamlines = function () {
  if (!this.velFieldData) return;
  if (this.streamlinesMesh) {
    this.scene.remove(this.streamlinesMesh);
    this.streamlinesMesh.geometry.dispose();
    this.streamlinesMesh.material.dispose();
  }

  var S = this.SIM_SIZE;
  var vel = this.velFieldData;
  var halfS = S / 2;
  var nSteps = 80;
  var stepSize = 0.7;

  // Seed points: grid upstream (density controlled)
  var seeds = [];
  var density = this.streamlineDensity || 5;
  var nY = density, nZ = density;
  for (var iy = 0; iy < nY; iy++) {
    for (var iz = 0; iz < nZ; iz++) {
      var sy = (iy / (nY-1) - 0.5) * S * 0.85;
      var sz = (iz / (nZ-1) - 0.5) * S * 0.85;
      seeds.push(-halfS * 0.95, sy, sz);
    }
  }

  var allPositions = [];
  var allColors = [];

  for (var si = 0; si < seeds.length; si += 3) {
    var px = seeds[si], py = seeds[si+1], pz = seeds[si+2];
    var linePositions = [px, py, pz];
    var lineSpeeds = [0];

    for (var step = 0; step < nSteps; step++) {
      var tx = (px + halfS) / S;
      var ty = (py + halfS) / S;
      var tz = (pz + halfS) / S;
      if (tx < 0.01 || tx >= 0.99 || ty < 0.01 || ty >= 0.99 || tz < 0.01 || tz >= 0.99) break;

      var fx = tx * (S-1), fy = ty * (S-1), fz = tz * (S-1);
      var ix = Math.max(0, Math.min(Math.floor(fx), S-2));
      var iy = Math.max(0, Math.min(Math.floor(fy), S-2));
      var iz = Math.max(0, Math.min(Math.floor(fz), S-2));
      var dx = fx - ix, dy = fy - iy, dz = fz - iz;

      function velAt(x,y,z) {
        var idx = (x + y*S + z*S*S)*3;
        return [vel[idx], vel[idx+1], vel[idx+2]];
      }
      var v000=velAt(ix,iy,iz), v100=velAt(ix+1,iy,iz);
      var v010=velAt(ix,iy+1,iz), v110=velAt(ix+1,iy+1,iz);
      var v001=velAt(ix,iy,iz+1), v101=velAt(ix+1,iy,iz+1);
      var v011=velAt(ix,iy+1,iz+1), v111=velAt(ix+1,iy+1,iz+1);

      var vx=0,vy=0,vz=0;
      for (var ci=0;ci<3;ci++) {
        var c00=v000[ci]*(1-dx)+v100[ci]*dx;
        var c01=v001[ci]*(1-dx)+v101[ci]*dx;
        var c10=v010[ci]*(1-dx)+v110[ci]*dx;
        var c11=v011[ci]*(1-dx)+v111[ci]*dx;
        var c0=c00*(1-dy)+c10*dy;
        var c1=c01*(1-dy)+c11*dy;
        var val=c0*(1-dz)+c1*dz;
        if (ci===0)vx=val; else if (ci===1)vy=val; else vz=val;
      }

      var spd = Math.sqrt(vx*vx+vy*vy+vz*vz);
      if (spd < 0.0001) break;

      px += vx / spd * stepSize;
      py += vy / spd * stepSize;
      pz += vz / spd * stepSize;

      if (Math.abs(px) > halfS*1.05 || Math.abs(py) > halfS*0.95 || Math.abs(pz) > halfS*0.95) break;

      linePositions.push(px, py, pz);
      lineSpeeds.push(spd);
    }

    if (linePositions.length >= 6) {
      var maxSpd = 0;
      for (var k=0;k<lineSpeeds.length;k++) if(lineSpeeds[k]>maxSpd) maxSpd=lineSpeeds[k];
      if (maxSpd < 0.001) maxSpd = 0.001;

      for (var li = 0; li < linePositions.length - 3; li += 3) {
        allPositions.push(
          linePositions[li], linePositions[li+1], linePositions[li+2],
          linePositions[li+3], linePositions[li+4], linePositions[li+5]
        );
        var t = lineSpeeds[Math.floor(li/3)+1] / maxSpd;
        // Rainbow: blue → cyan → green → yellow → red
        var r,g,b;
        if (t<0.25){r=0;g=t*4;b=1;}
        else if (t<0.5){r=0;g=1;b=1-(t-0.25)*4;}
        else if (t<0.75){r=(t-0.5)*4;g=1;b=0;}
        else{r=1;g=1-(t-0.75)*4;b=0;}
        allColors.push(r,g,b, r,g,b);
      }
    }
  }

  if (allPositions.length === 0) return;

  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(allPositions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(allColors, 3));

  var mat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    linewidth: 1
  });

  this.streamlinesMesh = new THREE.LineSegments(geo, mat);
  this.scene.add(this.streamlinesMesh);
  console.log('[App] Streamlines:', allPositions.length/6, 'segments');
};

// ---- 3D Velocity Field Cloud ----
WindTunnelApp.prototype.updateVelocityCloud = function () {
  if (!this.velFieldData) return;
  if (this.velCloudMesh) {
    this.scene.remove(this.velCloudMesh);
    this.velCloudMesh.geometry.dispose();
    this.velCloudMesh.material.dispose();
  }

  var S = this.SIM_SIZE;
  var vel = this.velFieldData;
  var halfS = S / 2;
  var step = 4; // sample every 4th cell
  var positions = [];
  var colors = [];

  for (var z = 1; z < S-1; z += step) {
    for (var y = 1; y < S-1; y += step) {
      for (var x = 1; x < S-1; x += step) {
        var idx = (x + y*S + z*S*S) * 3;
        var vx = vel[idx], vy = vel[idx+1], vz = vel[idx+2];
        var spd = Math.sqrt(vx*vx + vy*vy + vz*vz);
        if (spd < 0.005) continue; // skip slow cells

        var px = x - halfS, py = y - halfS, pz = z - halfS;
        positions.push(px, py, pz);

        var t = Math.min(spd * 5, 1);
        var r, g, b;
        if (t<0.25){r=0;g=t*4;b=1;}
        else if (t<0.5){r=0;g=1;b=1-(t-0.25)*4;}
        else if (t<0.75){r=(t-0.5)*4;g=1;b=0;}
        else{r=1;g=1-(t-0.75)*4;b=0;}
        colors.push(r, g, b);
      }
    }
  }

  if (positions.length === 0) return;

  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  var mat = new THREE.PointsMaterial({
    size: 0.8,
    vertexColors: true,
    transparent: true,
    opacity: 0.4,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true
  });

  this.velCloudMesh = new THREE.Points(geo, mat);
  this.scene.add(this.velCloudMesh);
  console.log('[App] Velocity cloud:', positions.length/3, 'points');
};

WindTunnelApp.prototype.showModelMesh = function (mesh) {
  // Remove old model visual
  if (this.modelVisual) {
    this.scene.remove(this.modelVisual);
    this.modelVisual.geometry.dispose();
    this.modelVisual.material.dispose();
  }

  var S = this.SIM_SIZE;
  var v = mesh.vertices, faces = mesh.faces, n = mesh.normals;

  // Compute bounds and scale (same as voxelization)
  var minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,minZ=Infinity,maxZ=-Infinity;
  for (var i=0;i<v.length;i+=3) {
    if(v[i]<minX)minX=v[i]; if(v[i]>maxX)maxX=v[i];
    if(v[i+1]<minY)minY=v[i+1]; if(v[i+1]>maxY)maxY=v[i+1];
    if(v[i+2]<minZ)minZ=v[i+2]; if(v[i+2]>maxZ)maxZ=v[i+2];
  }
  var sizeX=maxX-minX||1,sizeY=maxY-minY||1,sizeZ=maxZ-minZ||1;
  var scale = this.modelScale;
  var cellScale = (S*0.28)/Math.max(sizeX,sizeY,sizeZ)*scale;
  var cx=(minX+maxX)/2,cy=(minY+maxY)/2,cz=(minZ+maxZ)/2;

  // Create Three.js geometry
  var geo = new THREE.BufferGeometry();
  var positions = new Float32Array(faces.length * 3);
  var normals = new Float32Array(faces.length * 3);
  var hasVertexNormals = n && n.length >= v.length;

  for (var i = 0; i < faces.length; i += 3) {
    var i0=faces[i]*3, i1=faces[i+1]*3, i2=faces[i+2]*3;

    // Transform to grid space
    var ax=(v[i0]-cx)*cellScale+S/2, ay=(v[i0+1]-cy)*cellScale+S/2, az=(v[i0+2]-cz)*cellScale+S/2;
    var bx=(v[i1]-cx)*cellScale+S/2, by=(v[i1+1]-cy)*cellScale+S/2, bz=(v[i1+2]-cz)*cellScale+S/2;
    var ccx=(v[i2]-cx)*cellScale+S/2, ccy=(v[i2+1]-cy)*cellScale+S/2, ccz=(v[i2+2]-cz)*cellScale+S/2;

    // Center in scene (grid is centered at 0)
    positions[i*9+0]=ax-S/2;   positions[i*9+1]=ay-S/2;   positions[i*9+2]=az-S/2;
    positions[i*9+3]=bx-S/2;   positions[i*9+4]=by-S/2;   positions[i*9+5]=bz-S/2;
    positions[i*9+6]=ccx-S/2;  positions[i*9+7]=ccy-S/2;  positions[i*9+8]=ccz-S/2;

    // Use vertex normals from model if available, otherwise compute face normal
    if (hasVertexNormals) {
      normals[i*9+0]=n[i0];   normals[i*9+1]=n[i0+1];   normals[i*9+2]=n[i0+2];
      normals[i*9+3]=n[i1];   normals[i*9+4]=n[i1+1];   normals[i*9+5]=n[i1+2];
      normals[i*9+6]=n[i2];   normals[i*9+7]=n[i2+1];   normals[i*9+8]=n[i2+2];
    } else {
      var e1x=bx-ax,e1y=by-ay,e1z=bz-az;
      var e2x=ccx-ax,e2y=ccy-ay,e2z=ccz-az;
      var nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
      var len=Math.sqrt(nx*nx+ny*ny+nz*nz)||1;
      nx/=len; ny/=len; nz/=len;
      normals[i*9+0]=nx; normals[i*9+1]=ny; normals[i*9+2]=nz;
      normals[i*9+3]=nx; normals[i*9+4]=ny; normals[i*9+5]=nz;
      normals[i*9+6]=nx; normals[i*9+7]=ny; normals[i*9+8]=nz;
    }
  }

  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

  var mat = new THREE.MeshStandardMaterial({
    color: 0x6ec6ff,
    roughness: 0.25,
    metalness: 0.65,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
    envMapIntensity: 1.0,
    flatShading: false
  });

  this.modelVisual = new THREE.Mesh(geo, mat);
  this.scene.add(this.modelVisual);
  console.log('[App] Model visual added:', faces.length/3, 'triangles');
};

WindTunnelApp.prototype.createSphere = function () {
  var r = 0.28, latD = 20, lonD = 24;
  var verts = [], faces = [], norms = [];
  for (var i = 0; i <= latD; i++) {
    var theta = i * Math.PI / latD, sinT = Math.sin(theta), cosT = Math.cos(theta);
    for (var j = 0; j <= lonD; j++) {
      var phi = j * 2 * Math.PI / lonD;
      verts.push(r*Math.cos(phi)*sinT, r*cosT, r*Math.sin(phi)*sinT);
      norms.push(Math.cos(phi)*sinT, cosT, Math.sin(phi)*sinT);
    }
  }
  for (var ii = 0; ii < latD; ii++)
    for (var jj = 0; jj < lonD; jj++) {
      var a = ii*(lonD+1)+jj, b = a+lonD+1;
      faces.push(a,b,a+1,b,b+1,a+1);
    }
  return { vertices: verts, faces: faces, normals: norms };
};

WindTunnelApp.prototype.createCar = function () {
  var verts = [], faces = [], norms = [];

  // Generate smooth revolution surface
  function addRevSurface(profile, nSlices, cx, cy, cz, scaleX, scaleY, scaleZ) {
    var nP = profile.length / 2;
    var baseV = verts.length / 3;
    for (var s = 0; s <= nSlices; s++) {
      var angle = s * 2 * Math.PI / nSlices;
      var cosA = Math.cos(angle), sinA = Math.sin(angle);
      for (var p = 0; p < nP; p++) {
        var px = profile[p * 2];
        var py = profile[p * 2 + 1];
        // Revolution around Y axis
        var wx = cx + px * scaleX * cosA;
        var wy = cy + py * scaleY;
        var wz = cz + px * scaleZ * sinA;
        verts.push(wx, wy, wz);
        // Smooth normal
        var nx = py * cosA, ny = px, nz = py * sinA;
        var len = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
        norms.push(nx/len, ny/len, nz/len);
      }
    }
    for (var s = 0; s < nSlices; s++) {
      for (var p = 0; p < nP - 1; p++) {
        var a = baseV + s * nP + p;
        var b = a + nP;
        var c = a + 1;
        var d = b + 1;
        faces.push(a, b, c, c, b, d);
      }
    }
  }

  // Car body profile (cross-section in XY plane)
  // Points: [x (along length), y (height)]
  var bodyProfile = [
    [-0.28, -0.03],  // front bottom
    [-0.30, 0.00],   // front bumper lower
    [-0.32, 0.03],   // front bumper
    [-0.30, 0.06],   // hood front
    [-0.20, 0.07],   // hood
    [-0.12, 0.08],   // hood rear
    [-0.08, 0.12],   // windshield bottom
    [-0.02, 0.17],   // windshield top
    [0.02, 0.18],    // roof front
    [0.08, 0.18],    // roof
    [0.14, 0.17],    // roof rear
    [0.18, 0.14],    // rear window top
    [0.22, 0.10],    // rear window bottom
    [0.24, 0.08],    // trunk
    [0.28, 0.06],    // trunk rear
    [0.30, 0.04],    // rear bumper top
    [0.30, 0.01],    // rear bumper
    [0.28, -0.02],   // rear bottom
    [0.24, -0.04],   // rear underbody
    [0.00, -0.05],   // underbody
    [-0.24, -0.04],  // front underbody
  ];

  // Car width profile (half-width at each height level)
  var widthProfile = [
    0.16,  // bottom
    0.17,  // lower body
    0.18,  // body
    0.18,  // body
    0.18,  // body
    0.18,  // body
    0.17,  // windshield base
    0.15,  // windshield
    0.14,  // roof
    0.14,  // roof
    0.14,  // roof
    0.14,  // rear window
    0.15,  // rear window
    0.16,  // trunk
    0.17,  // trunk
    0.17,  // rear
    0.16,  // rear bumper
    0.16,  // rear bottom
    0.15,  // underbody
    0.16,  // underbody
    0.16,  // front underbody
  ];

  // Generate body mesh
  var nSlices = 24;
  var baseV = verts.length / 3;
  for (var s = 0; s <= nSlices; s++) {
    var angle = s * 2 * Math.PI / nSlices;
    var cosA = Math.cos(angle), sinA = Math.sin(angle);
    for (var p = 0; p < bodyProfile.length; p++) {
      var px = bodyProfile[p][0];
      var py = bodyProfile[p][1];
      var halfW = widthProfile[p];
      var wx = px;
      var wy = py;
      var wz = halfW * sinA;
      verts.push(wx, wy, wz);
      // Normal: approximate surface normal
      var nx = 0, ny = cosA, nz = sinA;
      if (p === 0 || p === bodyProfile.length - 1) { ny = -1; nz = 0; }
      var len = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
      norms.push(nx/len, ny/len, nz/len);
    }
  }
  var nP = bodyProfile.length;
  for (var s = 0; s < nSlices; s++) {
    for (var p = 0; p < nP - 1; p++) {
      var a = baseV + s * nP + p;
      var b = a + nP;
      var c = a + 1;
      var d = b + 1;
      faces.push(a, b, c, c, b, d);
    }
  }

  // Wheels (smooth octagonal)
  function addWheel(cx, cy, cz, r, w, segs) {
    var baseV2 = verts.length / 3;
    for (var s = 0; s <= segs; s++) {
      var angle = s * 2 * Math.PI / segs;
      var dx = Math.cos(angle) * r, dy = Math.sin(angle) * r;
      verts.push(cx - w/2, cy + dy, cz + dx); norms.push(-1, 0, 0);
      verts.push(cx + w/2, cy + dy, cz + dx); norms.push(1, 0, 0);
    }
    for (var s = 0; s < segs; s++) {
      var a = baseV2 + s * 2, b = a + 1, c = a + 2, d = a + 3;
      faces.push(a, c, b, b, c, d);
    }
    var centerL = verts.length / 3; verts.push(cx - w/2, cy, cz); norms.push(-1, 0, 0);
    var centerR = verts.length / 3; verts.push(cx + w/2, cy, cz); norms.push(1, 0, 0);
    for (var s = 0; s < segs; s++) {
      var a = baseV2 + s * 2, b = baseV2 + ((s + 1) % (segs + 1)) * 2;
      faces.push(centerL, b, a); faces.push(centerR, a + 1, b + 1);
    }
  }
  addWheel(-0.16, -0.04, 0.18, 0.035, 0.04, 12);
  addWheel(-0.16, -0.04, -0.18, 0.035, 0.04, 12);
  addWheel(0.16, -0.04, 0.18, 0.035, 0.04, 12);
  addWheel(0.16, -0.04, -0.18, 0.035, 0.04, 12);

  return { vertices: verts, faces: faces, normals: norms };
};

WindTunnelApp.prototype.createAirfoil = function () {
  var verts=[],faces=[],norms=[];
  var chord=0.5,thick=0.06,nPts=24,nSpan=16,span=0.4;
  for (var j=0;j<=nSpan;j++) {
    var z=(j/nSpan-0.5)*span;
    for (var i=0;i<=nPts;i++) {
      var t=i/nPts;
      var x=chord*(1-Math.cos(t*Math.PI))/2-chord*0.25;
      var y;
      if (t<=0.5) {
        var s=t*2;
        y=thick*(2.969*Math.sqrt(s)-1.26*s-3.516*s*s+2.843*s*s*s-1.015*s*s*s*s);
      } else {
        var s2=(1-t)*2;
        y=-thick*(2.969*Math.sqrt(s2)-1.26*s2-3.516*s2*s2+2.843*s2*s2*s2-1.015*s2*s2*s2*s2);
      }
      verts.push(x,y,z);
    }
  }
  for (var jj=0;jj<nSpan;jj++)
    for (var ii=0;ii<nPts;ii++) {
      var a=jj*(nPts+1)+ii,b=a+nPts+1;
      faces.push(a,b,a+1,b,b+1,a+1);
    }
  for (var k=0;k<verts.length/3;k++) {
    var tt=(k%(nPts+1))/nPts;
    var slope=Math.sin(tt*Math.PI)*0.5;
    var len=Math.sqrt(slope*slope+1);
    norms.push(slope/len,1/len,0);
  }
  return { vertices: verts, faces: faces, normals: norms };
};

// ---- Cube ----
WindTunnelApp.prototype.createCube = function () {
  var s = 0.28;
  var verts = [
    -s,-s,-s, s,-s,-s, s,s,-s, -s,s,-s,
    -s,-s, s, s,-s, s, s,s, s, -s,s, s
  ];
  var faces = [
    0,1,2, 0,2,3,  // front
    4,6,5, 4,7,6,  // back
    0,3,7, 0,7,4,  // left
    1,5,6, 1,6,2,  // right
    3,2,6, 3,6,7,  // top
    0,4,5, 0,5,1   // bottom
  ];
  var norms = [
    0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
    0,0,1, 0,0,1, 0,0,1, 0,0,1,
    -1,0,0, -1,0,0, -1,0,0, -1,0,0,
    1,0,0, 1,0,0, 1,0,0, 1,0,0,
    0,1,0, 0,1,0, 0,1,0, 0,1,0,
    0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0
  ];
  return { vertices: verts, faces: faces, normals: norms };
};

// ---- Cylinder ----
WindTunnelApp.prototype.createCylinder = function () {
  var r = 0.2, h = 0.5, segs = 24;
  var verts = [], faces = [], norms = [];
  // Side
  for (var i = 0; i <= segs; i++) {
    var angle = i * 2 * Math.PI / segs;
    var nx = Math.cos(angle), nz = Math.sin(angle);
    verts.push(nx*r, -h/2, nz*r); norms.push(nx, 0, nz);
    verts.push(nx*r,  h/2, nz*r); norms.push(nx, 0, nz);
  }
  for (var i = 0; i < segs; i++) {
    var a = i*2, b = a+1, c = a+2, d = a+3;
    faces.push(a, c, b, b, c, d);
  }
  // Top cap
  var topCenter = verts.length / 3;
  verts.push(0, h/2, 0); norms.push(0, 1, 0);
  for (var i = 0; i <= segs; i++) {
    var angle = i * 2 * Math.PI / segs;
    verts.push(Math.cos(angle)*r, h/2, Math.sin(angle)*r);
    norms.push(0, 1, 0);
  }
  for (var i = 0; i < segs; i++) {
    faces.push(topCenter, topCenter+1+i, topCenter+2+i);
  }
  // Bottom cap
  var botCenter = verts.length / 3;
  verts.push(0, -h/2, 0); norms.push(0, -1, 0);
  for (var i = 0; i <= segs; i++) {
    var angle = i * 2 * Math.PI / segs;
    verts.push(Math.cos(angle)*r, -h/2, Math.sin(angle)*r);
    norms.push(0, -1, 0);
  }
  for (var i = 0; i < segs; i++) {
    faces.push(botCenter, botCenter+2+i, botCenter+1+i);
  }
  return { vertices: verts, faces: faces, normals: norms };
};

// ---- F1 Race Car (improved) ----
WindTunnelApp.prototype.createF1Car = function () {
  var verts = [], faces = [], norms = [];

  // Smooth body surface using profile + revolution
  function addSmoothBody(profileX, profileY, profileW, nSlices) {
    var nP = profileX.length;
    var baseV = verts.length / 3;
    for (var s = 0; s <= nSlices; s++) {
      var angle = s * 2 * Math.PI / nSlices;
      var cosA = Math.cos(angle), sinA = Math.sin(angle);
      for (var p = 0; p < nP; p++) {
        var px = profileX[p];
        var py = profileY[p];
        var hw = profileW[p];
        var wy = py + hw * 0.3 * Math.max(0, cosA);
        var wz = hw * sinA;
        verts.push(px, wy, wz);
        var nx = 0, ny = cosA, nz = sinA;
        var len = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
        norms.push(nx/len, ny/len, nz/len);
      }
    }
    for (var s = 0; s < nSlices; s++) {
      for (var p = 0; p < nP - 1; p++) {
        var a = baseV + s * nP + p;
        var b = a + nP;
        var c = a + 1;
        var d = b + 1;
        faces.push(a, b, c, c, b, d);
      }
    }
  }

  // F1 body profile
  var bodyX = [-0.42, -0.38, -0.34, -0.28, -0.22, -0.16, -0.10, -0.04, 0.02, 0.08, 0.14, 0.20, 0.26, 0.30, 0.34, 0.36];
  var bodyY = [-0.02, -0.02, -0.02, -0.02, -0.02, -0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, -0.01, -0.02, -0.02];
  var bodyW = [0.03, 0.05, 0.06, 0.07, 0.08, 0.085, 0.09, 0.09, 0.09, 0.085, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02];
  addSmoothBody(bodyX, bodyY, bodyW, 16);

  // Nose cone (smooth taper)
  var noseX = [-0.50, -0.48, -0.46, -0.44, -0.42];
  var noseY = [-0.03, -0.025, -0.02, -0.02, -0.02];
  var noseW = [0.015, 0.025, 0.035, 0.04, 0.045];
  addSmoothBody(noseX, noseY, noseW, 12);

  // Cockpit (open top)
  var cockpitX = [-0.12, -0.08, -0.04, 0.0, 0.04];
  var cockpitY = [0.02, 0.03, 0.035, 0.035, 0.03];
  var cockpitW = [0.04, 0.045, 0.05, 0.05, 0.045];
  var cockpitBaseV = verts.length / 3;
  for (var p = 0; p < cockpitX.length; p++) {
    verts.push(cockpitX[p], cockpitY[p], -cockpitW[p]); norms.push(0, 0.7, -0.7);
    verts.push(cockpitX[p], cockpitY[p], cockpitW[p]); norms.push(0, 0.7, 0.7);
    verts.push(cockpitX[p], cockpitY[p] + 0.04, 0); norms.push(0, 1, 0);
  }
  for (var p = 0; p < cockpitX.length - 1; p++) {
    var a = cockpitBaseV + p * 3, b = a + 3;
    faces.push(a, b, a+1, a+1, b, b+1);
    faces.push(a+1, b+1, a+2, a+2, b+1, b+2);
  }

  // Headrest
  var headrestX = [0.04, 0.08, 0.12, 0.14];
  var headrestY = [0.03, 0.04, 0.04, 0.03];
  var headrestW = [0.035, 0.03, 0.025, 0.02];
  addSmoothBody(headrestX, headrestY, headrestW, 10);

  // Engine cover (smooth taper)
  var engineX = [0.14, 0.18, 0.22, 0.26, 0.30, 0.34];
  var engineY = [0.01, 0.02, 0.025, 0.02, 0.015, 0.01];
  var engineW = [0.06, 0.055, 0.05, 0.045, 0.04, 0.03];
  addSmoothBody(engineX, engineY, engineW, 12);

  // Airbox (above driver)
  var airboxX = [-0.04, -0.02, 0.0, 0.02, 0.04];
  var airboxY = [0.06, 0.07, 0.075, 0.07, 0.06];
  var airboxW = [0.02, 0.025, 0.03, 0.025, 0.02];
  addSmoothBody(airboxX, airboxY, airboxW, 8);

  // Front wing (multi-element, curved)
  function addWing(x, y, z, chord, thick, span, sweep) {
    var nSpan = 6;
    var baseVW = verts.length / 3;
    for (var sp = 0; sp <= nSpan; sp++) {
      var t = sp / nSpan;
      var zPos = z + (t - 0.5) * span;
      var xOffset = t * span * sweep;
      var chordScale = 1 - t * 0.3;
      verts.push(x + xOffset - chord*chordScale/2, y, zPos); norms.push(0, -1, 0);
      verts.push(x + xOffset + chord*chordScale/2, y, zPos); norms.push(0, -1, 0);
      verts.push(x + xOffset - chord*chordScale/2, y + thick, zPos); norms.push(0, 1, 0);
      verts.push(x + xOffset + chord*chordScale/2, y + thick, zPos); norms.push(0, 1, 0);
    }
    for (var sp = 0; sp < nSpan; sp++) {
      var a = baseVW + sp * 4, b = a + 4;
      faces.push(a, b, a+1, a+1, b, b+1); // bottom
      faces.push(a+2, a+3, b+2, a+3, b+3, b+2); // top
      faces.push(a, a+2, b, a+2, b+2, b); // leading edge
      faces.push(a+1, b+1, a+3, a+3, b+1, b+3); // trailing edge
    }
  }
  addWing(-0.44, -0.04, 0, 0.06, 0.008, 0.44, 0.02);
  addWing(-0.42, -0.05, 0, 0.05, 0.006, 0.40, 0.015);
  addWing(-0.40, -0.055, 0, 0.04, 0.005, 0.36, 0.01);

  // Rear wing (tall, multi-element)
  addWing(0.34, 0.06, 0, 0.04, 0.008, 0.32, -0.01);
  addWing(0.33, 0.08, 0, 0.035, 0.007, 0.28, -0.008);
  addWing(0.32, 0.10, 0, 0.03, 0.006, 0.24, -0.006);

  // Endplates
  function addEndplate(x, y, z, h, w) {
    var baseVE = verts.length / 3;
    verts.push(x - w/2, y, z); norms.push(0, 0, -1);
    verts.push(x + w/2, y, z); norms.push(0, 0, -1);
    verts.push(x - w/2, y + h, z); norms.push(0, 0, -1);
    verts.push(x + w/2, y + h, z); norms.push(0, 0, -1);
    faces.push(baseVE, baseVE+1, baseVE+2, baseVE+2, baseVE+1, baseVE+3);
  }
  addEndplate(-0.44, -0.04, 0.22, 0.04, 0.008);
  addEndplate(-0.44, -0.04, -0.22, 0.04, 0.008);
  addEndplate(0.34, 0.06, 0.16, 0.10, 0.008);
  addEndplate(0.34, 0.06, -0.16, 0.10, 0.008);

  // Rear wing pillar
  verts.push(0.34, 0.01, -0.005); norms.push(0, 0, -1);
  verts.push(0.34, 0.01, 0.005); norms.push(0, 0, 1);
  verts.push(0.34, 0.06, -0.005); norms.push(0, 0, -1);
  verts.push(0.34, 0.06, 0.005); norms.push(0, 0, 1);
  var pv = verts.length / 3 - 4;
  faces.push(pv, pv+1, pv+2, pv+2, pv+1, pv+3);

  // Wheels (smooth)
  function addWheel(cx, cy, cz, r, w, segs) {
    var baseV2 = verts.length / 3;
    for (var s = 0; s <= segs; s++) {
      var angle = s * 2 * Math.PI / segs;
      var dx = Math.cos(angle) * r, dy = Math.sin(angle) * r;
      verts.push(cx - w/2, cy + dy, cz + dx); norms.push(-1, 0, 0);
      verts.push(cx + w/2, cy + dy, cz + dx); norms.push(1, 0, 0);
    }
    for (var s = 0; s < segs; s++) {
      var a = baseV2 + s * 2, b = a + 1, c = a + 2, d = a + 3;
      faces.push(a, c, b, b, c, d);
    }
    var centerL = verts.length / 3; verts.push(cx - w/2, cy, cz); norms.push(-1, 0, 0);
    var centerR = verts.length / 3; verts.push(cx + w/2, cy, cz); norms.push(1, 0, 0);
    for (var s = 0; s < segs; s++) {
      var a = baseV2 + s * 2, b = baseV2 + ((s + 1) % (segs + 1)) * 2;
      faces.push(centerL, b, a); faces.push(centerR, a + 1, b + 1);
    }
  }
  addWheel(-0.24, -0.05, 0.14, 0.04, 0.06, 12);
  addWheel(-0.24, -0.05, -0.14, 0.04, 0.06, 12);
  addWheel(0.26, -0.05, 0.15, 0.05, 0.08, 12);
  addWheel(0.26, -0.05, -0.15, 0.05, 0.08, 12);

  // Sidepods (smooth)
  var spX = [-0.02, 0.02, 0.06, 0.10, 0.14, 0.18];
  var spY = [-0.02, -0.025, -0.03, -0.03, -0.025, -0.02];
  var spW = [0.05, 0.06, 0.065, 0.065, 0.06, 0.05];
  addSmoothBody(spX, spY, spW, 10);
  // Mirror for other side
  var spX2 = spX.slice();
  var spY2 = spY.slice();
  var spW2 = spW.slice();
  addSmoothBody(spX2, spY2, spW2, 10);

  // Floor/diffuser
  var floorVerts = verts.length / 3;
  var floorX = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.36];
  var floorW = [0.12, 0.14, 0.16, 0.17, 0.17, 0.16, 0.14, 0.10];
  for (var p = 0; p < floorX.length; p++) {
    verts.push(floorX[p], -0.05, -floorW[p]); norms.push(0, -1, 0);
    verts.push(floorX[p], -0.05, floorW[p]); norms.push(0, -1, 0);
  }
  for (var p = 0; p < floorX.length - 1; p++) {
    var a = floorVerts + p * 2, b = a + 2;
    faces.push(a, b, a+1, a+1, b, b+1);
  }

  return { vertices: verts, faces: faces, normals: norms };
};

// ---- Bridge (improved with arch and details) ----
WindTunnelApp.prototype.createBridge = function () {
  var verts = [], faces = [], norms = [];

  // Bridge deck (smooth road surface)
  var deckX = [-0.36, -0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36];
  var deckW = [0.10, 0.11, 0.115, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.115, 0.11, 0.10, 0.09];
  var deckBaseV = verts.length / 3;
  for (var p = 0; p < deckX.length; p++) {
    var x = deckX[p];
    var hw = deckW[p];
    verts.push(x, 0.0, -hw); norms.push(0, 1, 0);
    verts.push(x, 0.0, hw); norms.push(0, 1, 0);
    verts.push(x, -0.02, -hw * 0.9); norms.push(0, -1, 0);
    verts.push(x, -0.02, hw * 0.9); norms.push(0, -1, 0);
  }
  for (var p = 0; p < deckX.length - 1; p++) {
    var a = deckBaseV + p * 4, b = a + 4;
    faces.push(a, b, a+1, a+1, b, b+1); // top surface
    faces.push(a+2, a+3, b+2, a+3, b+3, b+2); // bottom surface
    faces.push(a, a+2, b, a+2, b+2, b); // left edge
    faces.push(a+1, b+1, a+3, a+3, b+1, b+3); // right edge
  }

  // Side rails (smooth curve on top)
  var railX = [-0.36, -0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36];
  function addRail(zPos) {
    var railBaseV = verts.length / 3;
    var hw = 0.12;
    for (var p = 0; p < railX.length; p++) {
      var x = railX[p];
      verts.push(x, 0.0, zPos - 0.006); norms.push(0, 0, -1);
      verts.push(x, 0.0, zPos + 0.006); norms.push(0, 0, 1);
      verts.push(x, 0.04, zPos - 0.006); norms.push(0, 0, -1);
      verts.push(x, 0.04, zPos + 0.006); norms.push(0, 0, 1);
    }
    for (var p = 0; p < railX.length - 1; p++) {
      var a = railBaseV + p * 4, b = a + 4;
      faces.push(a, b, a+2, a+2, b, b+2); // front
      faces.push(a+1, a+3, b+1, a+3, b+3, b+1); // back
      faces.push(a+2, b+2, a+3, a+3, b+2, b+3); // top
    }
  }
  addRail(0.12);
  addRail(-0.12);

  // Main arch (smooth parabolic curve)
  var archN = 16;
  var archSpan = 0.56;
  var archHeight = 0.18;
  var archWidth = 0.06;
  for (var side = -1; side <= 1; side += 2) {
    var archBaseV = verts.length / 3;
    for (var i = 0; i <= archN; i++) {
      var t = i / archN;
      var x = -archSpan/2 + t * archSpan;
      var y = archHeight * (1 - (2*t - 1) * (2*t - 1)); // Parabola
      var z = side * archWidth / 2;
      verts.push(x - 0.015, y, z - 0.015); norms.push(-1, 0, -1);
      verts.push(x + 0.015, y, z - 0.015); norms.push(1, 0, -1);
      verts.push(x - 0.015, y, z + 0.015); norms.push(-1, 0, 1);
      verts.push(x + 0.015, y, z + 0.015); norms.push(1, 0, 1);
    }
    for (var i = 0; i < archN; i++) {
      var a = archBaseV + i * 4, b = a + 4;
      faces.push(a, b, a+1, a+1, b, b+1);
      faces.push(a+2, a+3, b+2, a+3, b+3, b+2);
      faces.push(a, a+2, b, a+2, b+2, b);
      faces.push(a+1, b+1, a+3, a+3, b+1, b+3);
    }
  }

  // Arch supports (tapered pillars)
  function addPillar(x, z) {
    var pillarBaseV = verts.length / 3;
    var pillarH = 0.20;
    var rBottom = 0.025;
    var rTop = 0.018;
    var nSegs = 8;
    for (var s = 0; s <= nSegs; s++) {
      var angle = s * 2 * Math.PI / nSegs;
      var cosA = Math.cos(angle), sinA = Math.sin(angle);
      verts.push(x + rBottom * cosA, -pillarH, z + rBottom * sinA);
      norms.push(cosA, 0, sinA);
      verts.push(x + rTop * cosA, 0.0, z + rTop * sinA);
      norms.push(cosA, 0, sinA);
    }
    for (var s = 0; s < nSegs; s++) {
      var a = pillarBaseV + s * 2, b = a + 2, c = a + 1, d = a + 3;
      faces.push(a, b, c, c, b, d);
    }
  }
  addPillar(-0.20, 0);
  addPillar(0.0, 0);
  addPillar(0.20, 0);

  // Suspender cables (from arch to deck)
  var cablePositions = [-0.22, -0.14, -0.06, 0.02, 0.10, 0.18, 0.26];
  for (var c = 0; c < cablePositions.length; c++) {
    var cx = cablePositions[c];
    var t = (cx + archSpan/2) / archSpan;
    var archY = archHeight * (1 - (2*t - 1) * (2*t - 1));
    var cableBaseV = verts.length / 3;
    var r = 0.004;
    for (var s = 0; s <= 6; s++) {
      var angle = s * 2 * Math.PI / 6;
      var cosA = Math.cos(angle), sinA = Math.sin(angle);
      verts.push(cx + r * cosA, 0.04, 0.06 + r * sinA); norms.push(cosA, 0, sinA);
      verts.push(cx + r * cosA, archY, 0.06 + r * sinA); norms.push(cosA, 0, sinA);
      verts.push(cx + r * cosA, 0.04, -0.06 + r * sinA); norms.push(cosA, 0, sinA);
      verts.push(cx + r * cosA, archY, -0.06 + r * sinA); norms.push(cosA, 0, sinA);
    }
    for (var s = 0; s < 6; s++) {
      var a = cableBaseV + s * 4, b = a + 4;
      faces.push(a, b, a+1, a+1, b, b+1);
      faces.push(a+2, a+3, b+2, a+3, b+3, b+2);
    }
  }

  return { vertices: verts, faces: faces, normals: norms };
};

// ---- Airplane (improved with smooth fuselage) ----
WindTunnelApp.prototype.createAirplane = function () {
  var verts = [], faces = [], norms = [];

  // Smooth fuselage: generate as revolution surface along X axis
  function addSmoothFuselage(profileX, profileR, nSlices, cx, cy, cz, scaleY, scaleZ) {
    var nP = profileX.length;
    var baseV = verts.length / 3;
    for (var s = 0; s <= nSlices; s++) {
      var angle = s * 2 * Math.PI / nSlices;
      var cosA = Math.cos(angle), sinA = Math.sin(angle);
      for (var p = 0; p < nP; p++) {
        var px = profileX[p];
        var r = profileR[p];
        var wx = cx + px;
        var wy = cy + r * scaleY * cosA;
        var wz = cz + r * scaleZ * sinA;
        verts.push(wx, wy, wz);
        // Normal: radial from centerline
        var nx = 0, ny = cosA, nz = sinA;
        var len = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
        norms.push(nx/len, ny/len, nz/len);
      }
    }
    for (var s = 0; s < nSlices; s++) {
      for (var p = 0; p < nP - 1; p++) {
        var a = baseV + s * nP + p;
        var b = a + nP;
        var c = a + 1;
        var d = b + 1;
        faces.push(a, b, c, c, b, d);
      }
    }
  }

  // Fuselage profile (X position, radius)
  var fuselageX = [-0.42, -0.38, -0.32, -0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24, 0.30, 0.34, 0.36];
  var fuselageR = [0.015, 0.03, 0.045, 0.055, 0.06, 0.062, 0.063, 0.062, 0.058, 0.05, 0.04, 0.025, 0.01];
  addSmoothFuselage(fuselageX, fuselageR, 16, 0, 0, 0, 1.0, 1.0);

  // Cockpit canopy (dome on top)
  var canopyX = [-0.20, -0.14, -0.08, -0.02, 0.04];
  var canopyR = [0.02, 0.03, 0.035, 0.03, 0.015];
  var canopyBaseV = verts.length / 3;
  var nCanopySlices = 12;
  for (var s = 0; s <= nCanopySlices; s++) {
    var angle = s * Math.PI / nCanopySlices; // Only top half
    var cosA = Math.cos(angle);
    for (var p = 0; p < canopyX.length; p++) {
      var px = canopyX[p];
      var r = canopyR[p];
      var wy = 0.06 + r * Math.max(0, cosA);
      var wz = r * Math.sin(angle);
      verts.push(px, wy, wz);
      norms.push(0, cosA, Math.sin(angle));
    }
  }
  var nCP = canopyX.length;
  for (var s = 0; s < nCanopySlices; s++) {
    for (var p = 0; p < nCP - 1; p++) {
      var a = canopyBaseV + s * nCP + p;
      var b = a + nCP;
      var c = a + 1;
      var d = b + 1;
      faces.push(a, b, c, c, b, d);
    }
  }

  // Main wings (swept, with airfoil cross-section)
  function addWing(zPos, spanDir) {
    var wingX = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12];
    var wingChord = [0.12, 0.14, 0.15, 0.14, 0.12, 0.08];
    var wingThick = [0.015, 0.018, 0.02, 0.018, 0.015, 0.01];
    var sweepAngle = 0.3; // radians
    var nSpan = 8;
    var spanLength = 0.22;
    var baseV2 = verts.length / 3;
    for (var sp = 0; sp <= nSpan; sp++) {
      var t = sp / nSpan;
      var z = zPos + spanDir * t * spanLength;
      var sweepOffset = t * spanLength * Math.tan(sweepAngle);
      for (var p = 0; p < wingX.length; p++) {
        var px = wingX[p] + sweepOffset;
        var chord = wingChord[p] * (1 - t * 0.4);
        var thick = wingThick[p] * (1 - t * 0.5);
        var py = thick * 0.5;
        verts.push(px, py, z);
        norms.push(0, 1, 0);
      }
      for (var p = wingX.length - 1; p >= 0; p--) {
        var px = wingX[p] + sweepOffset;
        var chord = wingChord[p] * (1 - t * 0.4);
        var thick = wingThick[p] * (1 - t * 0.5);
        var py = -thick * 0.5;
        verts.push(px, py, z);
        norms.push(0, -1, 0);
      }
    }
    var nWP = wingX.length * 2;
    for (var sp = 0; sp < nSpan; sp++) {
      for (var p = 0; p < nWP - 1; p++) {
        var a = baseV2 + sp * nWP + p;
        var b = a + nWP;
        var c = a + 1;
        var d = b + 1;
        faces.push(a, b, c, c, b, d);
      }
      // Close the loop
      var a = baseV2 + sp * nWP + nWP - 1;
      var b = a + nWP;
      var c = baseV2 + sp * nWP;
      var d = c + nWP;
      faces.push(a, b, c, c, b, d);
    }
  }
  addWing(0.04, 1);
  addWing(-0.04, -1);

  // Tail vertical stabilizer
  var tailFinX = [0.24, 0.28, 0.32, 0.34];
  var tailFinH = [0.0, 0.06, 0.10, 0.08];
  var tailFinBaseV = verts.length / 3;
  for (var p = 0; p < tailFinX.length; p++) {
    verts.push(tailFinX[p], 0.04, 0.0); norms.push(0, 0, 1);
    verts.push(tailFinX[p], 0.04 + tailFinH[p], 0.0); norms.push(0, 0, 1);
    verts.push(tailFinX[p], 0.04, 0.006); norms.push(0, 0, -1);
    verts.push(tailFinX[p], 0.04 + tailFinH[p], 0.006); norms.push(0, 0, -1);
  }
  for (var p = 0; p < tailFinX.length - 1; p++) {
    var a = tailFinBaseV + p * 4, b = a + 4, c = a + 1, d = a + 5;
    faces.push(a, b, c, c, b, d);
    a += 2; b += 2; c = a + 1; d = a + 5;
    faces.push(a, b, c, c, b, d);
  }

  // Tail horizontal stabilizers
  function addHStab(zPos, spanDir) {
    var hstabX = [0.26, 0.30, 0.34];
    var hstabChord = [0.06, 0.05, 0.03];
    var nSpan2 = 4;
    var spanLen = 0.10;
    var baseV3 = verts.length / 3;
    for (var sp = 0; sp <= nSpan2; sp++) {
      var t = sp / nSpan2;
      var z = zPos + spanDir * t * spanLen;
      for (var p = 0; p < hstabX.length; p++) {
        var px = hstabX[p] + t * 0.03;
        var py = 0.02 + 0.005 * (1 - t);
        verts.push(px, py, z);
        norms.push(0, 1, 0);
        verts.push(px, py - 0.008, z);
        norms.push(0, -1, 0);
      }
    }
    var nHP = hstabX.length * 2;
    for (var sp = 0; sp < nSpan2; sp++) {
      for (var p = 0; p < nHP - 1; p++) {
        var a = baseV3 + sp * nHP + p;
        var b = a + nHP;
        var c = a + 1;
        var d = b + 1;
        faces.push(a, b, c, c, b, d);
      }
    }
  }
  addHStab(0.02, 1);
  addHStab(-0.02, -1);

  // Engine nacelles (smooth cylinders under wings)
  function addEngine(cx, cy, cz) {
    var eX = [-0.06, -0.02, 0.02, 0.06, 0.08];
    var eR = [0.025, 0.03, 0.032, 0.03, 0.02];
    var nESlices = 10;
    var baseVE = verts.length / 3;
    for (var s = 0; s <= nESlices; s++) {
      var angle = s * 2 * Math.PI / nESlices;
      var cosA = Math.cos(angle), sinA = Math.sin(angle);
      for (var p = 0; p < eX.length; p++) {
        var px = eX[p];
        var r = eR[p];
        verts.push(cx + px, cy + r * cosA, cz + r * sinA);
        norms.push(0, cosA, sinA);
      }
    }
    var nEP = eX.length;
    for (var s = 0; s < nESlices; s++) {
      for (var p = 0; p < nEP - 1; p++) {
        var a = baseVE + s * nEP + p;
        var b = a + nEP;
        var c = a + 1;
        var d = b + 1;
        faces.push(a, b, c, c, b, d);
      }
    }
  }
  addEngine(-0.02, -0.05, 0.14);
  addEngine(-0.02, -0.05, -0.14);

  return { vertices: verts, faces: faces, normals: norms };
};

// ---- OBJ Parser ----
WindTunnelApp.prototype.parseOBJ = function (text) {
  var positions = [], normals = [], faceTokens = [];
  var lines = text.split('\n');
  for (var li = 0; li < lines.length; li++) {
    var line = lines[li].trim();
    if (!line || line[0] === '#') continue;
    var parts = line.split(/\s+/);
    if (parts[0] === 'v') {
      positions.push(parseFloat(parts[1])||0, parseFloat(parts[2])||0, parseFloat(parts[3])||0);
    } else if (parts[0] === 'vn') {
      normals.push(parseFloat(parts[1])||0, parseFloat(parts[2])||0, parseFloat(parts[3])||0);
    } else if (parts[0] === 'f') {
      faceTokens.push(parts.slice(1));
    }
  }

  var uniqueMap = {};
  var outVerts = [], outNorms = [], outFaces = [];
  var nextIdx = 0;

  function processVertex(token) {
    if (uniqueMap[token] !== undefined) return uniqueMap[token];
    var sp = token.split('/');
    var pi = (parseInt(sp[0]) - 1) * 3;
    var ni = sp.length > 2 && sp[2] ? (parseInt(sp[2]) - 1) * 3 : -1;
    var idx = nextIdx++;
    outVerts.push(positions[pi]||0, positions[pi+1]||0, positions[pi+2]||0);
    outNorms.push(ni>=0?(normals[ni]||0):0, ni>=0?(normals[ni+1]||0):0, ni>=0?(normals[ni+2]||0):0);
    uniqueMap[token] = idx;
    return idx;
  }

  for (var fi = 0; fi < faceTokens.length; fi++) {
    var tokens = faceTokens[fi];
    if (tokens.length < 3) continue;
    var v0 = processVertex(tokens[0]);
    for (var ti = 1; ti < tokens.length - 1; ti++) {
      outFaces.push(v0, processVertex(tokens[ti]), processVertex(tokens[ti+1]));
    }
  }

  // Auto-compute normals if none provided
  var hasNormals = normals.length > 0;
  if (!hasNormals) {
    outNorms = new Array(outVerts.length).fill(0);
    for (var f = 0; f < outFaces.length; f += 3) {
      var i0 = outFaces[f]*3, i1 = outFaces[f+1]*3, i2 = outFaces[f+2]*3;
      var e1x = outVerts[i1]-outVerts[i0], e1y = outVerts[i1+1]-outVerts[i0+1], e1z = outVerts[i1+2]-outVerts[i0+2];
      var e2x = outVerts[i2]-outVerts[i0], e2y = outVerts[i2+1]-outVerts[i0+1], e2z = outVerts[i2+2]-outVerts[i0+2];
      var nx = e1y*e2z-e1z*e2y, ny = e1z*e2x-e1x*e2z, nz = e1x*e2y-e1y*e2x;
      outNorms[i0]+=nx; outNorms[i0+1]+=ny; outNorms[i0+2]+=nz;
      outNorms[i1]+=nx; outNorms[i1+1]+=ny; outNorms[i1+2]+=nz;
      outNorms[i2]+=nx; outNorms[i2+1]+=ny; outNorms[i2+2]+=nz;
    }
    for (var n = 0; n < outNorms.length; n += 3) {
      var len = Math.sqrt(outNorms[n]*outNorms[n]+outNorms[n+1]*outNorms[n+1]+outNorms[n+2]*outNorms[n+2]);
      if (len > 0) { outNorms[n]/=len; outNorms[n+1]/=len; outNorms[n+2]/=len; }
    }
  }

  return { vertices: outVerts, faces: outFaces, normals: outNorms };
};

// ---- STL Parser ----
WindTunnelApp.prototype.parseSTL = function (buffer) {
  var verts = [], faces = [], norms = [];
  var dv = new DataView(buffer);
  if (buffer.byteLength < 84) return { vertices: verts, faces: faces, normals: norms, error: '文件太小，不是有效的STL文件' };

  var isASCII = String.fromCharCode(dv.getUint8(0),dv.getUint8(1),dv.getUint8(2),dv.getUint8(3),dv.getUint8(4)).toLowerCase() === 'solid';

  if (isASCII) {
    var text = '';
    var arr = new Uint8Array(buffer);
    for (var ci = 0; ci < arr.length; ci++) text += String.fromCharCode(arr[ci]);
    var nx=0,ny=0,nz=0,vCount=0;
    var tlines = text.split('\n');
    for (var li = 0; li < tlines.length; li++) {
      var p = tlines[li].trim().split(/\s+/);
      if (p[0]==='facet'&&p[1]==='normal') { nx=parseFloat(p[2])||0;ny=parseFloat(p[3])||0;nz=parseFloat(p[4])||0; }
      else if (p[0]==='vertex') {
        verts.push(parseFloat(p[1])||0,parseFloat(p[2])||0,parseFloat(p[3])||0);
        norms.push(nx,ny,nz);
        vCount++;
        if (vCount%3===0) { var bv=verts.length/3-3; faces.push(bv,bv+1,bv+2); }
      }
    }
  } else {
    var nTriangles = dv.getUint32(80, true);
    if (nTriangles === 0) return { vertices: verts, faces: faces, normals: norms, error: 'STL文件不包含任何三角面' };
    if (nTriangles > 2000000) return { vertices: verts, faces: faces, normals: norms, error: 'STL文件三角面过多 (' + nTriangles + ')，请简化模型后重试' };
    var offset = 84;
    var requiredSize = 84 + nTriangles * 50;
    if (buffer.byteLength < requiredSize) return { vertices: verts, faces: faces, normals: norms, error: 'STL文件数据不完整，期望 ' + nTriangles + ' 个三角面' };
    for (var ti = 0; ti < nTriangles; ti++) {
      var snx=dv.getFloat32(offset,true),sny=dv.getFloat32(offset+4,true),snz=dv.getFloat32(offset+8,true);
      offset+=12;
      var bv2=verts.length/3;
      for (var sv=0;sv<3;sv++) {
        var px=dv.getFloat32(offset,true),py=dv.getFloat32(offset+4,true),pz=dv.getFloat32(offset+8,true);
        if (isNaN(px)||isNaN(py)||isNaN(pz)) return { vertices: verts, faces: faces, normals: norms, error: 'STL文件第 ' + (ti+1) + ' 个三角面包含无效坐标' };
        verts.push(px,py,pz);
        norms.push(snx,sny,snz);
        offset+=12;
      }
      faces.push(bv2,bv2+1,bv2+2);
      offset+=2;
    }
  }

  // Validate result
  if (verts.length === 0) return { vertices: verts, faces: faces, normals: norms, error: 'STL文件未解析到任何顶点' };
  if (faces.length === 0) return { vertices: verts, faces: faces, normals: norms, error: 'STL文件未解析到任何三角面' };

  // Check for degenerate model (all points same)
  var minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,minZ=Infinity,maxZ=-Infinity;
  for (var i=0;i<verts.length;i+=3) {
    if(verts[i]<minX)minX=verts[i]; if(verts[i]>maxX)maxX=verts[i];
    if(verts[i+1]<minY)minY=verts[i+1]; if(verts[i+1]>maxY)maxY=verts[i+1];
    if(verts[i+2]<minZ)minZ=verts[i+2]; if(verts[i+2]>maxZ)maxZ=verts[i+2];
  }
  var sizeX=maxX-minX, sizeY=maxY-minY, sizeZ=maxZ-minZ;
  if (sizeX < 1e-10 && sizeY < 1e-10 && sizeZ < 1e-10) return { vertices: verts, faces: faces, normals: norms, error: '模型尺寸为零，请检查文件' };

  return { vertices: verts, faces: faces, normals: norms };
};

WindTunnelApp.prototype.loadFile = function (file) {
  var self = this;
  document.getElementById('status-text').textContent = '加载 ' + file.name + '...';

  // File size check
  if (file.size > 50 * 1024 * 1024) {
    document.getElementById('status-text').textContent = '❌ 文件过大 (' + (file.size/1024/1024).toFixed(1) + 'MB)，请使用小于50MB的文件';
    return;
  }
  if (file.size === 0) {
    document.getElementById('status-text').textContent = '❌ 文件为空';
    return;
  }

  if (file.name.toLowerCase().endsWith('.obj')) {
    var reader = new FileReader();
    reader.onload = function (e) {
      try {
        var mesh = self.parseOBJ(e.target.result);
        if (!mesh || !mesh.vertices || mesh.vertices.length === 0) {
          document.getElementById('status-text').textContent = '❌ OBJ文件解析失败：未找到有效顶点';
          return;
        }
        if (!mesh.faces || mesh.faces.length === 0) {
          document.getElementById('status-text').textContent = '❌ OBJ文件解析失败：未找到有效面';
          return;
        }
        self.currentModelMesh = mesh;
        self.currentModel = file.name;
        var triCount = mesh.faces.length / 3;
        document.getElementById('status-text').textContent = '✅ ' + file.name + ' 已加载 (' + triCount + ' 个三角面)';
        self.voxelizeAndLoad(mesh);
      } catch (err) {
        document.getElementById('status-text').textContent = '❌ OBJ解析错误: ' + err.message;
        console.error('[OBJ Parse Error]', err);
      }
    };
    reader.readAsText(file);
  } else if (file.name.toLowerCase().endsWith('.stl')) {
    var reader2 = new FileReader();
    reader2.onload = function (e) {
      try {
        var mesh = self.parseSTL(e.target.result);
        if (mesh.error) {
          document.getElementById('status-text').textContent = '❌ STL错误: ' + mesh.error;
          return;
        }
        if (!mesh.vertices || mesh.vertices.length === 0) {
          document.getElementById('status-text').textContent = '❌ STL文件解析失败：未找到有效顶点';
          return;
        }
        self.currentModelMesh = mesh;
        self.currentModel = file.name;
        var triCount = mesh.faces.length / 3;
        document.getElementById('status-text').textContent = '✅ ' + file.name + ' 已加载 (' + triCount + ' 个三角面)';
        self.showModelMesh(mesh);
        self.voxelizeAndLoad(mesh);
      } catch (err) {
        document.getElementById('status-text').textContent = '❌ STL解析错误: ' + err.message;
        console.error('[STL Parse Error]', err);
      }
    };
    reader2.readAsArrayBuffer(file);
  } else {
    document.getElementById('status-text').textContent = '❌ 不支持的格式，请使用 OBJ 或 STL';
  }
};

// ---- Voxelization ----
WindTunnelApp.prototype.voxelizeAndLoad = function (mesh) {
  var self = this;
  var S = this.SIM_SIZE;
  var obstacle = new Uint8Array(S * S * S);
  var v = mesh.vertices, faces = mesh.faces;

  // Bounds
  var minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,minZ=Infinity,maxZ=-Infinity;
  for (var i=0;i<v.length;i+=3) {
    if(v[i]<minX)minX=v[i]; if(v[i]>maxX)maxX=v[i];
    if(v[i+1]<minY)minY=v[i+1]; if(v[i+1]>maxY)maxY=v[i+1];
    if(v[i+2]<minZ)minZ=v[i+2]; if(v[i+2]>maxZ)maxZ=v[i+2];
  }
  var sizeX=maxX-minX||1,sizeY=maxY-minY||1,sizeZ=maxZ-minZ||1;
  var scale = this.modelScale;
  var cellScale = (S*0.28)/Math.max(sizeX,sizeY,sizeZ)*scale;
  var cx=(minX+maxX)/2,cy=(minY+maxY)/2,cz=(minZ+maxZ)/2;

  var gMinX=Math.floor(S/2-(maxX-cx)*cellScale-1), gMaxX=Math.ceil(S/2-(minX-cx)*cellScale+1);
  var gMinY=Math.floor(S/2-(maxY-cy)*cellScale-1), gMaxY=Math.ceil(S/2-(minY-cy)*cellScale+1);
  var gMinZ=Math.floor(S/2-(maxZ-cz)*cellScale-1), gMaxZ=Math.ceil(S/2-(minZ-cz)*cellScale+1);
  gMinX=Math.max(1,gMinX); gMaxX=Math.min(S-2,gMaxX);
  gMinY=Math.max(1,gMinY); gMaxY=Math.min(S-2,gMaxY);
  gMinZ=Math.max(1,gMinZ); gMaxZ=Math.min(S-2,gMaxZ);

  // Pre-compute triangle bounding boxes in grid space for acceleration
  var nTri = faces.length / 3;
  var triBBox = new Float32Array(nTri * 6); // [minX,minY,minZ,maxX,maxY,maxZ] per tri
  for (var ti = 0; ti < nTri; ti++) {
    var i0 = faces[ti*3]*3, i1 = faces[ti*3+1]*3, i2 = faces[ti*3+2]*3;
    var tMinX = Math.min(v[i0], v[i1], v[i2]);
    var tMinY = Math.min(v[i0+1], v[i1+1], v[i2+1]);
    var tMinZ = Math.min(v[i0+2], v[i1+2], v[i2+2]);
    var tMaxX = Math.max(v[i0], v[i1], v[i2]);
    var tMaxY = Math.max(v[i0+1], v[i1+1], v[i2+1]);
    var tMaxZ = Math.max(v[i0+2], v[i1+2], v[i2+2]);
    // Expand by cell size for safety
    var cellWorld = 1.0 / cellScale;
    triBBox[ti*6]   = tMinX - cellWorld;
    triBBox[ti*6+1] = tMinY - cellWorld;
    triBBox[ti*6+2] = tMinZ - cellWorld;
    triBBox[ti*6+3] = tMaxX + cellWorld;
    triBBox[ti*6+4] = tMaxY + cellWorld;
    triBBox[ti*6+5] = tMaxZ + cellWorld;
  }

  // Build cell list
  var cells = [];
  for (var gz=gMinZ;gz<=gMaxZ;gz++)
    for (var gy=gMinY;gy<=gMaxY;gy++)
      for (var gx=gMinX;gx<=gMaxX;gx++)
        cells.push(gx,gy,gz);

  var eps=1e-9, solidCount=0, idx=0;
  var CHUNK = 4000; // smaller chunks for better UI responsiveness
  var totalCells = cells.length / 3;

  document.getElementById('status-text').textContent = '体素化中... (0%)';

  function processChunk() {
    var startTime = performance.now();
    var end = Math.min(idx + CHUNK * 3, cells.length);
    for (var ci = idx; ci < end; ci += 3) {
      var gx=cells[ci], gy=cells[ci+1], gz=cells[ci+2];
      var wx=(gx-S/2)/cellScale+cx, wy=(gy-S/2)/cellScale+cy, wz=(gz-S/2)/cellScale+cz;
      var hits=0;
      for (var fi=0;fi<nTri;fi++) {
        // Bounding box pre-filter: skip triangles far from this cell
        var bb = fi * 6;
        if (wx < triBBox[bb] || wx > triBBox[bb+3] ||
            wy < triBBox[bb+1] || wy > triBBox[bb+4] ||
            wz < triBBox[bb+2] || wz > triBBox[bb+5]) continue;

        var i0=faces[fi*3]*3,i1=faces[fi*3+1]*3,i2=faces[fi*3+2]*3;
        var ax=v[i0],ay=v[i0+1],az=v[i0+2];
        var e1x=v[i1]-ax,e1y=v[i1+1]-ay,e1z=v[i1+2]-az;
        var e2x=v[i2]-ax,e2y=v[i2+1]-ay,e2z=v[i2+2]-az;
        var hhy=1*e2z-0*e2y, hhz=0*e2x-1*e2z, hhx=1*e2y-0*e2x;
        var a=e1x*hhx+e1y*hhy+e1z*hhz;
        if (a>-eps&&a<eps) continue;
        var f=1/a;
        var sx2=wx-ax,sy2=wy-ay,sz2=wz-az;
        var u=f*(sx2*hhx+sy2*hhy+sz2*hhz);
        if (u<0||u>1) continue;
        var qx=sy2*e1z-sz2*e1y,qy=sz2*e1x-sx2*e1z,qz=sx2*e1y-sy2*e1x;
        var vv=f*(1*qx+0*qy+0*qz);
        if (vv>=0&&u+vv<=1&&f*(e2x*qx+e2y*qy+e2z*qz)>eps) hits++;
      }
      if (hits%2===1) { obstacle[gx+gy*S+gz*S*S]=1; solidCount++; }
    }
    idx = end;

    var pct = Math.round(idx/cells.length*100);
    var elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    document.getElementById('status-text').textContent =
      '体素化中... ' + pct + '% (' + solidCount + ' 个固体网格, ' + elapsed + 's/chunk)';

    if (idx < cells.length) {
      setTimeout(processChunk, 0);
    } else {
      document.getElementById('status-text').textContent = '✅ 模型已加载 (' + solidCount + ' 个固体网格)';
      self.worker.postMessage({ type: 'loadObstacle', data: obstacle.buffer }, [obstacle.buffer]);
    }
  }

  processChunk();
};

// ---- Worker communication ----
WindTunnelApp.prototype.startWorker = function () {
  var S = this.SIM_SIZE;
  this.worker.postMessage({
    type: 'init',
    sizeX: S, sizeY: S, sizeZ: S,
    velocity: this.velocity,
    viscosity: this.viscosity,
    stepsPerFrame: this.stepsPerFrame
  });
};

WindTunnelApp.prototype.sendStep = function () {
  this.worker.postMessage({ type: 'step' });
};

// ---- UI ----
WindTunnelApp.prototype.initUI = function () {
  var self = this;
  var $ = function (id) { return document.getElementById(id); };

  $('btn-start').addEventListener('click', function () {
    console.log('[App] Start clicked, running=', self.running);
    self.running = true;
    $('btn-start').style.display = 'none';
    $('btn-pause').style.display = '';
    document.body.classList.add('simulating');
    $('status-text').textContent = '模拟运行中...';
    self.worker.postMessage({ type: 'requestData' });
  });

  $('btn-pause').addEventListener('click', function () {
    self.running = false;
    $('btn-pause').style.display = 'none';
    $('btn-start').style.display = '';
    document.body.classList.remove('simulating');
    $('status-text').textContent = '已暂停';
  });

  $('btn-reset').addEventListener('click', function () {
    self.running = false;
    self.stepCount = 0;
    $('btn-pause').style.display = 'none';
    $('btn-start').style.display = '';
    document.body.classList.remove('simulating');
    $('step-count').textContent = '步数: 0';
    $('status-text').textContent = '重置中...';
    self.resetSimulation();
  });

  // Grid buttons
  var gridBtns = document.querySelectorAll('.grid-btn');
  for (var gi = 0; gi < gridBtns.length; gi++) {
    (function (btn) {
      btn.addEventListener('click', function () {
        for (var j = 0; j < gridBtns.length; j++) gridBtns[j].classList.remove('active');
        btn.classList.add('active');
        self.SIM_SIZE = parseInt(btn.getAttribute('data-grid'));
        self.resetSimulation();
      });
    })(gridBtns[gi]);
  }

  // Speed / viscosity
  $('ctrl-speed').addEventListener('input', function (e) {
    self.velocity = parseFloat(e.target.value);
    $('val-speed').textContent = self.velocity.toFixed(3);
    self.worker.postMessage({ type: 'setVelocity', value: self.velocity, viscosity: self.viscosity });
    self.updateColorBar();
  });
  $('ctrl-viscosity').addEventListener('input', function (e) {
    self.viscosity = parseFloat(e.target.value);
    $('val-viscosity').textContent = self.viscosity.toFixed(3);
    self.worker.postMessage({ type: 'setVelocity', value: self.velocity, viscosity: self.viscosity });
  });

  // Model buttons
  var modelBtns = document.querySelectorAll('.model-btn');
  for (var mi = 0; mi < modelBtns.length; mi++) {
    (function (btn) {
      btn.addEventListener('click', function () {
        for (var j = 0; j < modelBtns.length; j++) modelBtns[j].classList.remove('active');
        btn.classList.add('active');
        self.loadModel(btn.getAttribute('data-model'));
      });
    })(modelBtns[mi]);
  }

  // Auto-detect button
  $('btn-auto').addEventListener('click', function () {
    self.autoProfile();
  });

  // Log panel
  var logEntries = [];
  var origLog = console.log;
  var origErr = console.error;
  var origWarn = console.warn;
  function captureLog(type, args) {
    var msg = Array.prototype.slice.call(args).map(function(a) {
      return typeof a === 'object' ? JSON.stringify(a) : String(a);
    }).join(' ');
    var ts = new Date().toLocaleTimeString();
    logEntries.push('[' + ts + '] [' + type + '] ' + msg);
    if (logEntries.length > 500) logEntries.shift();
  }
  console.log = function() { captureLog('INFO', arguments); origLog.apply(console, arguments); };
  console.error = function() { captureLog('ERROR', arguments); origErr.apply(console, arguments); };
  console.warn = function() { captureLog('WARN', arguments); origWarn.apply(console, arguments); };

  $('btn-log').addEventListener('click', function () {
    var panel = document.getElementById('log-panel');
    var content = document.getElementById('log-content');
    content.textContent = logEntries.join('\n');
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    content.scrollTop = content.scrollHeight;
  });
  $('btn-log-copy').addEventListener('click', function () {
    var text = logEntries.join('\n');
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(function() { origLog('日志已复制到剪贴板'); });
    } else {
      var ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
  });
  $('btn-log-close').addEventListener('click', function () {
    document.getElementById('log-panel').style.display = 'none';
  });

  // File upload
  $('btn-upload').addEventListener('click', function () { $('file-input').click(); });
  $('file-input').addEventListener('change', function (e) {
    if (e.target.files.length > 0) {
      for (var j = 0; j < modelBtns.length; j++) modelBtns[j].classList.remove('active');
      self.loadFile(e.target.files[0]);
    }
  });

  // Streamline density
  var ctrlStreamlines = document.getElementById('ctrl-streamlines');
  if (ctrlStreamlines) {
    ctrlStreamlines.addEventListener('input', function (e) {
      self.streamlineDensity = parseInt(e.target.value);
      var label = document.getElementById('val-streamlines');
      if (label) label.textContent = self.streamlineDensity;
    });
    ctrlStreamlines.addEventListener('change', function () {
      if (self.velFieldData) self.updateStreamlines();
    });
  }

  // Scale
  $('ctrl-scale').addEventListener('input', function (e) {
    self.modelScale = parseFloat(e.target.value);
    $('val-scale').textContent = self.modelScale.toFixed(1);
  });
  $('ctrl-scale').addEventListener('change', function () {
    if (self.currentModelMesh) self.voxelizeAndLoad(self.currentModelMesh);
  });

  // Visualization
  $('ctrl-visual').addEventListener('change', function (e) {
    var mode = e.target.value;
    if (self.particleSystem) self.particleSystem.visible = (mode === 'particles' || mode === 'both');
    if (self.sliceMesh) self.sliceMesh.visible = (mode === 'slice' || mode === 'both');
  });
  $('ctrl-colormap').addEventListener('change', function (e) {
    var cmap = parseInt(e.target.value);
    if (self.sliceMesh) self.sliceMesh.material.uniforms.uColormap.value = cmap;
    self.updateColorBar();
  });
  $('ctrl-slice').addEventListener('input', function (e) {
    var pos = parseFloat(e.target.value);
    $('val-slice').textContent = pos.toFixed(2);
    if (self.sliceMesh) {
      self.sliceMesh.material.uniforms.uSlicePos.value = pos;
      var S = self.SIM_SIZE;
      if (self.sliceAxis === 0) { self.sliceMesh.position.x = (pos - 0.5) * S; self.sliceMesh.position.y = 0; self.sliceMesh.position.z = 0; self.sliceMesh.rotation.set(0, Math.PI/2, 0); }
      else if (self.sliceAxis === 1) { self.sliceMesh.position.y = (pos - 0.5) * S; self.sliceMesh.position.x = 0; self.sliceMesh.position.z = 0; self.sliceMesh.rotation.set(Math.PI/2, 0, 0); }
      else { self.sliceMesh.position.z = (pos - 0.5) * S; self.sliceMesh.position.x = 0; self.sliceMesh.position.y = 0; self.sliceMesh.rotation.set(0, 0, 0); }
    }
  });

  // Slice axis selection
  $('ctrl-slice-axis').addEventListener('change', function (e) {
    self.sliceAxis = parseInt(e.target.value);
    if (self.sliceMesh) {
      self.sliceMesh.material.uniforms.uSliceAxis.value = self.sliceAxis;
      // Reposition slice on new axis
      var pos = parseFloat($('ctrl-slice').value);
      var S = self.SIM_SIZE;
      if (self.sliceAxis === 0) { self.sliceMesh.position.set((pos-0.5)*S, 0, 0); self.sliceMesh.rotation.set(0, Math.PI/2, 0); }
      else if (self.sliceAxis === 1) { self.sliceMesh.position.set(0, (pos-0.5)*S, 0); self.sliceMesh.rotation.set(Math.PI/2, 0, 0); }
      else { self.sliceMesh.position.set(0, 0, (pos-0.5)*S); self.sliceMesh.rotation.set(0, 0, 0); }
    }
  });

  // Slice dragging with mouse
  var sliceDragging = false;
  var sliceRaycaster = new THREE.Raycaster();
  var slicePlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);

  self.renderer.domElement.addEventListener('mousedown', function (e) {
    if (!self.sliceMesh || !self.sliceMesh.visible) return;
    var rect = self.renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    sliceRaycaster.setFromCamera(mouse, self.camera);
    var intersects = sliceRaycaster.intersectObject(self.sliceMesh);
    if (intersects.length > 0) {
      sliceDragging = true;
      self.controls.enabled = false;
    }
  });

  self.renderer.domElement.addEventListener('mousemove', function (e) {
    if (!sliceDragging) return;
    var rect = self.renderer.domElement.getBoundingClientRect();
    var mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    sliceRaycaster.setFromCamera(mouse, self.camera);
    var intersection = new THREE.Vector3();
    sliceRaycaster.ray.intersectPlane(slicePlane, intersection);
    if (intersection) {
      var S = self.SIM_SIZE;
      var pos = Math.max(0.1, Math.min(0.9, intersection.x / S + 0.5));
      self.sliceMesh.position.x = (pos - 0.5) * S;
      self.sliceMesh.material.uniforms.uSlicePos.value = pos;
      var slider = document.getElementById('ctrl-slice');
      slider.value = pos;
      document.getElementById('val-slice').textContent = pos.toFixed(2);
    }
  });

  self.renderer.domElement.addEventListener('mouseup', function () {
    if (sliceDragging) {
      sliceDragging = false;
      self.controls.enabled = true;
    }
  });
  $('ctrl-particles').addEventListener('input', function (e) {
    var count = parseInt(e.target.value);
    $('val-particles').textContent = (count/1000).toFixed(0) + 'K';
  });
  $('ctrl-particles').addEventListener('change', function (e) {
    self.initParticles(parseInt(e.target.value));
  });
  $('ctrl-pspeed').addEventListener('input', function (e) {
    var speed = parseFloat(e.target.value);
    $('val-pspeed').textContent = speed.toFixed(1);
    if (self.particleSystem) self.particleSystem.material.uniforms.uSpeed.value = speed;
  });
};

// ---- Auto-profiling ----
WindTunnelApp.prototype.autoProfile = function () {
  var self = this;
  var statusEl = document.getElementById('status-text');
  if (statusEl) statusEl.textContent = '正在检测GPU性能...';

  // Run profiling after a short delay to let the UI render
  setTimeout(function () {
    var profile = profileGPU();
    self.gpuProfile = profile;
    var presetName = pickPreset(profile.score);
    var preset = QUALITY_PRESETS[presetName];
    self.qualityPreset = presetName;
    self.applyPreset(preset);

    // Update UI
    var autoBtn = document.getElementById('btn-auto');
    if (autoBtn) autoBtn.classList.add('active');
    self.updatePresetDisplay();

    // Update status
    var gpuShort = profile.gpu.length > 40 ? profile.gpu.substring(0, 40) + '...' : profile.gpu;
    var info = '已检测: ' + gpuShort + ' | 基准: ' + profile.elapsed.toFixed(1) + 'ms | ';
    info += preset.label + ' (' + preset.grid + '³, ' + (preset.particles/1000) + 'K粒子)';
    if (statusEl) statusEl.textContent = info;

    // Reload model with new grid
    self.resetSimulation();
  }, 100);
};

WindTunnelApp.prototype.applyPreset = function (preset) {
  this.SIM_SIZE = preset.grid;
  this.particleCount = preset.particles;
  this.stepsPerFrame = preset.stepsPerFrame;
};

WindTunnelApp.prototype.updatePresetDisplay = function () {
  var display = document.getElementById('quality-display');
  if (display) {
    var preset = QUALITY_PRESETS[this.qualityPreset];
    display.textContent = preset.label;
  }
  // Sync grid buttons
  var gridBtns = document.querySelectorAll('.grid-btn');
  for (var i = 0; i < gridBtns.length; i++) {
    var btn = gridBtns[i];
    btn.classList.toggle('active', parseInt(btn.getAttribute('data-grid')) === this.SIM_SIZE);
  }
  // Sync particle slider
  var pSlider = document.getElementById('ctrl-particles');
  var pLabel = document.getElementById('val-particles');
  if (pSlider) pSlider.value = this.particleCount;
  if (pLabel) pLabel.textContent = (this.particleCount / 1000).toFixed(0) + 'K';
};

WindTunnelApp.prototype.resetSimulation = function () {
  this.running = false;
  this.stepCount = 0;
  document.getElementById('btn-pause').style.display = 'none';
  document.getElementById('btn-start').style.display = '';
  document.body.classList.remove('simulating');
  this.worker.postMessage({ type: 'terminate' });
  this.initVelocityField();
  this.initParticles(this.particleCount);
  this.initSlice();
  this.initWorker();
  this.loadModel(this.currentModel);
};

// ---- Camera views ----
WindTunnelApp.prototype.setCameraView = function (view) {
  var S = this.SIM_SIZE, d = S * 1.4;
  var positions = {
    side: [d, S*0.2, 0],
    top: [0, d, 0.01],
    front: [0, S*0.2, d],
    free: [d*0.6, d*0.4, d*0.6]
  };
  var pos = positions[view] || positions.free;
  var cam = this.camera, ctrl = this.controls;
  var start = { x: cam.position.x, y: cam.position.y, z: cam.position.z };
  var end = { x: pos[0], y: pos[1], z: pos[2] };
  var duration = 800, startTime = performance.now();
  (function animateView() {
    var t = Math.min((performance.now() - startTime) / duration, 1);
    var ease = 1 - Math.pow(1 - t, 3);
    cam.position.set(start.x+(end.x-start.x)*ease, start.y+(end.y-start.y)*ease, start.z+(end.z-start.z)*ease);
    ctrl.update();
    if (t < 1) requestAnimationFrame(animateView);
  })();
};

// ---- Animation loop ----
WindTunnelApp.prototype.animate = function () {
  var self = this;
  requestAnimationFrame(function () { self.animate(); });
  this.controls.update();

  this.frameCount++;
  var now = performance.now();
  if (now - this.lastFpsTime >= 1000) {
    document.getElementById('fps-counter').textContent = 'FPS: ' + this.frameCount;
    this.frameCount = 0;
    this.lastFpsTime = now;
  }
  document.getElementById('step-count').textContent = '步数: ' + (this.stepCount * this.stepsPerFrame).toLocaleString();
  var re = (this.velocity * (this.SIM_SIZE * 0.28) / this.viscosity).toFixed(0);
  document.getElementById('re-counter').textContent = 'Re: ' + re;

  if (this.particleSystem && this.particleSystem.visible && this.velFieldData) {
    // Reduce particle update frequency when slice is also visible to avoid lag
    if (this.sliceMesh && this.sliceMesh.visible) {
      this.frameCount % 2 === 0 && this.updateParticles();
    } else {
      this.updateParticles();
    }
  }

  this.renderer.render(this.scene, this.camera);
};

// ---- Boot ----
try {
  if (typeof THREE === 'undefined') {
    console.error('Three.js not loaded');
    var el = document.getElementById('status-text');
    if (el) el.textContent = '错误: Three.js 未加载，请检查网络';
  } else if (typeof OrbitControls === 'undefined') {
    console.error('OrbitControls not loaded');
    var el = document.getElementById('status-text');
    if (el) el.textContent = '错误: OrbitControls 未加载，请刷新重试';
  } else {
    window.app = new WindTunnelApp();
  }
} catch (e) {
  console.error('App init error:', e);
  var el = document.getElementById('status-text');
  if (el) el.textContent = '初始化错误: ' + e.message;
}
