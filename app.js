// ============================================================
//  Wind Tunnel CFD Simulator — Lattice Boltzmann Method (D3Q19)
//  Compatible with Chrome 56+, Firefox 51+, Safari 15+, Edge 79+
// ============================================================

(function () {
'use strict';

// ---- Compatibility check ----
(function checkCompat() {
  var canvas = document.createElement('canvas');
  var gl = canvas.getContext('webgl2');
  if (!gl) {
    var el = document.getElementById('compat-warn');
    if (el) el.style.display = 'flex';
    var app = document.getElementById('app');
    if (app) {
      var vp = document.getElementById('viewport');
      if (vp) vp.style.display = 'none';
      var tb = document.getElementById('toolbar');
      if (tb) tb.style.display = 'none';
    }
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
  var fs = 'varying float vVal; void main(){ gl_FragColor=vec4(vVal,vVal,vVal,1); }';
  var prog = gl.createProgram();
  function mkShader(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    gl.attachShader(prog, s);
    return s;
  }
  var vsh = mkShader(gl.VERTEX_SHADER, vs);
  var fsh = mkShader(gl.FRAGMENT_SHADER, fs);
  gl.linkProgram(prog);
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

  this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
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
  this.scene.add(new THREE.AmbientLight(0x4488cc, 0.6));
  var d1 = new THREE.DirectionalLight(0xffffff, 0.8);
  d1.position.set(50, 80, 60);
  this.scene.add(d1);
  var d2 = new THREE.DirectionalLight(0x4488cc, 0.3);
  d2.position.set(-40, 20, -30);
  this.scene.add(d2);
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
  var positions = new Float32Array(count * 3);
  var ages = new Float32Array(count);
  var randoms = new Float32Array(count);
  for (var i = 0; i < count; i++) {
    positions[i*3]   = (Math.random() - 0.5) * S * 1.1;
    positions[i*3+1] = (Math.random() - 0.5) * S * 0.95;
    positions[i*3+2] = (Math.random() - 0.5) * S * 0.95;
    ages[i] = Math.random() * 3.0;
    randoms[i] = Math.random();
  }

  var geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('aAge', new THREE.BufferAttribute(ages, 1));
  geo.setAttribute('aRandom', new THREE.BufferAttribute(randoms, 1));

  var vertShader = [
    'uniform sampler3D uVelocityField;',
    'uniform float uSize;',
    'uniform float uSimSize;',
    'uniform float uSpeed;',
    'uniform int uColormap;',
    'varying float vSpeed;',
    'varying float vAge;',
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
    '  vec3 pos = position;',
    '  float age = aAge;',
    '  for (int step = 0; step < 2; step++) {',
    '    vec3 tc = (pos + uSimSize * 0.5) / uSimSize;',
    '    tc = clamp(tc, 0.005, 0.995);',
    '    vec3 vel = texture(uVelocityField, tc).rgb;',
    '    pos += vel * uSpeed * 15.0;',
    '    age += 0.016;',
    '    if (age > 3.0 || pos.x > uSimSize*0.55 || pos.x < -uSimSize*0.55 ||',
    '        abs(pos.y) > uSimSize*0.5 || abs(pos.z) > uSimSize*0.5) {',
    '      pos.x = -uSimSize*0.55 - aRandom*uSimSize*0.1;',
    '      pos.y = (aRandom*2.0-1.0)*uSimSize*0.45;',
    '      pos.z = (fract(aRandom*7.0)*2.0-1.0)*uSimSize*0.45;',
    '      age = 0.0;',
    '    }',
    '  }',
    '  vec3 tc2 = clamp((pos + uSimSize*0.5) / uSimSize, 0.005, 0.995);',
    '  float spd = length(texture(uVelocityField, tc2).rgb);',
    '  vSpeed = spd; vAge = age;',
    '  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);',
    '  gl_PointSize = uSize * (220.0 / -mvPos.z);',
    '  gl_Position = projectionMatrix * mvPos;',
    '}'
  ].join('\n');

  var fragShader = [
    'varying float vSpeed;',
    'varying float vAge;',
    'uniform int uColormap;',
    'uniform sampler2D uSprite;',
    'vec3 getColor(float t, int cmap) {',
    '  // Smooth step interpolation for each palette',
    '  if (cmap == 1) {',
    '    t = clamp(t, 0.0, 1.0);',
    '    if (t<0.25) return mix(vec3(0,.2,1),vec3(.3,.65,1),smoothstep(0.0,0.25,t));',
    '    if (t<0.5)  return mix(vec3(.3,.65,1),vec3(.9,.9,.9),smoothstep(0.25,0.5,t));',
    '    if (t<0.75) return mix(vec3(.9,.9,.9),vec3(1,.4,.2),smoothstep(0.5,0.75,t));',
    '    return mix(vec3(1,.4,.2),vec3(.7,0,0),smoothstep(0.75,1.0,t));',
    '  } else if (cmap == 2) {',
    '    float s = sin(t*18.0)*0.5+0.5;',
    '    return mix(vec3(0,.3,.8),vec3(.8,.1,0),s);',
    '  } else {',
    '    t = clamp(t, 0.0, 1.0);',
    '    if (t<0.2) return mix(vec3(.1,.1,.9),vec3(0,.6,1),smoothstep(0.0,0.2,t));',
    '    if (t<0.4) return mix(vec3(0,.6,1),vec3(0,1,.5),smoothstep(0.2,0.4,t));',
    '    if (t<0.6) return mix(vec3(0,1,.5),vec3(.8,1,0),smoothstep(0.4,0.6,t));',
    '    if (t<0.8) return mix(vec3(.8,1,0),vec3(1,.5,0),smoothstep(0.6,0.8,t));',
    '    return mix(vec3(1,.5,0),vec3(1,.05,0),smoothstep(0.8,1.0,t));',
    '  }',
    '}',
    'void main() {',
    '  vec4 sprite = texture2D(uSprite, gl_PointCoord);',
    '  if (sprite.a < 0.01) discard;',
    '  float fadeAge = vAge > 2.5 ? (3.0 - vAge) * 2.0 : 1.0;',
    '  float t = clamp(vSpeed * 5.0, 0.0, 1.0);',
    '  vec3 color = getColor(t, uColormap);',
    '  float glow = 0.7 + 0.3 * sprite.a;',
    '  float alpha = sprite.a * fadeAge * 0.9 * glow;',
    '  gl_FragColor = vec4(color * glow, alpha);',
    '}'
  ].join('\n');

  var spriteTex = this.makeParticleTexture();

  var mat = new THREE.ShaderMaterial({
    vertexShader: vertShader,
    fragmentShader: fragShader,
    uniforms: {
      uVelocityField: { value: this.velTexture },
      uSprite: { value: spriteTex },
      uSize: { value: 3.0 },
      uSimSize: { value: S },
      uSpeed: { value: 1.0 },
      uColormap: { value: 0 }
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });

  this.particleSystem = new THREE.Points(geo, mat);
  this.scene.add(this.particleSystem);
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
    '  vec3 tc = vec3(uSlicePos, vUv.x, vUv.y);',
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
      uColormap: { value: 0 }
    },
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false
  });

  this.sliceMesh = new THREE.Mesh(geo, mat);
  this.sliceMesh.rotation.y = Math.PI / 2;
  this.sliceMesh.position.x = 0;
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
    '    obstacle=new Uint8Array(N);',
    '    fIn=new Float32Array(N*Q);fOut=new Float32Array(N*Q);',
    '    for (var i=0;i<N;i++) { rho[i]=1; for (var q=0;q<Q;q++) fIn[q*N+i]=w[q]; }',
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
    if (d.type === 'ready') self.onWorkerReady();
    else if (d.type === 'obstacleLoaded') self.onObstacleLoaded();
    else if (d.type === 'stepDone') self.onStepDone();
    else if (d.type === 'data') self.onVelocityData(d.buffer);
  };
};

WindTunnelApp.prototype.onWorkerReady = function () {
  document.getElementById('status-text').textContent = '模拟引擎就绪';
};

WindTunnelApp.prototype.onObstacleLoaded = function () {
  document.getElementById('status-text').textContent = '模型已加载 - 点击开始模拟';
};

WindTunnelApp.prototype.onStepDone = function () {
  this.stepCount++;
  if (this.running) this.worker.postMessage({ type: 'requestData' });
};

WindTunnelApp.prototype.onVelocityData = function (buffer) {
  this.velocityData = new Float32Array(buffer);
  var texData = this.velTexture.image.data;
  var N = this.SIM_SIZE * this.SIM_SIZE * this.SIM_SIZE;
  for (var i = 0; i < N; i++) {
    texData[i*4]   = this.velocityData[i*3];
    texData[i*4+1] = this.velocityData[i*3+1];
    texData[i*4+2] = this.velocityData[i*3+2];
    texData[i*4+3] = 0;
  }
  this.velTexture.needsUpdate = true;
};

// ---- Model loading ----
WindTunnelApp.prototype.loadModel = function (name) {
  var mesh;
  if (name === 'sphere') mesh = this.createSphere();
  else if (name === 'car') mesh = this.createCar();
  else if (name === 'airfoil') mesh = this.createAirfoil();
  this.currentModelMesh = mesh;
  this.currentModel = name;
  this.voxelizeAndLoad(mesh);
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
  function addBox(cx,cy,cz,sx,sy,sz) {
    var dx=sx/2,dy=sy/2,dz=sz/2, v=verts.length/3;
    var pts=[[cx-dx,cy-dy,cz-dz],[cx+dx,cy-dy,cz-dz],[cx+dx,cy+dy,cz-dz],[cx-dx,cy+dy,cz-dz],
             [cx-dx,cy-dy,cz+dz],[cx+dx,cy-dy,cz+dz],[cx+dx,cy+dy,cz+dz],[cx-dx,cy+dy,cz+dz]];
    for (var p=0;p<pts.length;p++) verts.push(pts[p][0],pts[p][1],pts[p][2]);
    var bf=[[v,v+1,v+2,v,v+2,v+3],[v+4,v+6,v+5,v+4,v+7,v+6],
            [v,v+3,v+7,v,v+7,v+4],[v+1,v+5,v+6,v+1,v+6,v+2],
            [v+3,v+2,v+6,v+3,v+6,v+7],[v,v+4,v+5,v,v+5,v+1]];
    var bn=[[0,0,-1],[0,0,1],[-1,0,0],[1,0,0],[0,1,0],[0,-1,0]];
    for (var fi=0;fi<bf.length;fi++) {
      faces.push(bf[fi][0],bf[fi][1],bf[fi][2],bf[fi][3],bf[fi][4],bf[fi][5]);
      for (var ni=0;ni<6;ni++) norms.push(bn[fi][0],bn[fi][1],bn[fi][2]);
    }
  }
  addBox(0,-0.02,0,0.5,0.12,0.16);
  addBox(0.02,0.08,0,0.22,0.1,0.14);
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
  if (buffer.byteLength < 84) return { vertices: verts, faces: faces, normals: norms };

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
    var offset = 84;
    for (var ti = 0; ti < nTriangles; ti++) {
      var snx=dv.getFloat32(offset,true),sny=dv.getFloat32(offset+4,true),snz=dv.getFloat32(offset+8,true);
      offset+=12;
      var bv2=verts.length/3;
      for (var sv=0;sv<3;sv++) {
        verts.push(dv.getFloat32(offset,true),dv.getFloat32(offset+4,true),dv.getFloat32(offset+8,true));
        norms.push(snx,sny,snz);
        offset+=12;
      }
      faces.push(bv2,bv2+1,bv2+2);
      offset+=2;
    }
  }
  return { vertices: verts, faces: faces, normals: norms };
};

WindTunnelApp.prototype.loadFile = function (file) {
  var self = this;
  document.getElementById('status-text').textContent = '加载 ' + file.name + '...';
  if (file.name.toLowerCase().endsWith('.obj')) {
    var reader = new FileReader();
    reader.onload = function (e) {
      var mesh = self.parseOBJ(e.target.result);
      self.currentModelMesh = mesh;
      self.currentModel = file.name;
      self.voxelizeAndLoad(mesh);
    };
    reader.readAsText(file);
  } else if (file.name.toLowerCase().endsWith('.stl')) {
    var reader2 = new FileReader();
    reader2.onload = function (e) {
      var mesh = self.parseSTL(e.target.result);
      self.currentModelMesh = mesh;
      self.currentModel = file.name;
      self.voxelizeAndLoad(mesh);
    };
    reader2.readAsArrayBuffer(file);
  } else {
    document.getElementById('status-text').textContent = '不支持的格式，请使用 OBJ 或 STL';
  }
};

// ---- Voxelization ----
WindTunnelApp.prototype.voxelizeAndLoad = function (mesh) {
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

  var eps=1e-9;
  var solidCount=0;

  for (var gz=gMinZ;gz<=gMaxZ;gz++) {
    for (var gy=gMinY;gy<=gMaxY;gy++) {
      for (var gx=gMinX;gx<=gMaxX;gx++) {
        var wx=(gx-S/2)/cellScale+cx, wy=(gy-S/2)/cellScale+cy, wz=(gz-S/2)/cellScale+cz;
        if (wx<minX-eps||wx>maxX+eps||wy<minY-eps||wy>maxY+eps||wz<minZ-eps||wz>maxZ+eps) continue;

        var hits=0;
        for (var fi=0;fi<faces.length;fi+=3) {
          var i0=faces[fi]*3,i1=faces[fi+1]*3,i2=faces[fi+2]*3;
          var ax=v[i0],ay=v[i0+1],az=v[i0+2];
          var e1x=v[i1]-ax,e1y=v[i1+1]-ay,e1z=v[i1+2]-az;
          var e2x=v[i2]-ax,e2y=v[i2+1]-ay,e2z=v[i2+2]-az;
          var hx=e2z,e2y_=e2x; // reuse vars carefully
          // Ray-triangle (Möller–Trumbore)
          var hhy=1*e2z-0*e2y, hhz=0*e2x-1*e2z, hhx=1*e2y-0*e2x; // dir=(1,0,0)
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
    }
  }

  document.getElementById('status-text').textContent = '模型已加载 (' + solidCount + ' 个固体网格)';
  this.worker.postMessage({ type: 'loadObstacle', data: obstacle.buffer }, [obstacle.buffer]);
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

  // File upload
  $('btn-upload').addEventListener('click', function () { $('file-input').click(); });
  $('file-input').addEventListener('change', function (e) {
    if (e.target.files.length > 0) {
      for (var j = 0; j < modelBtns.length; j++) modelBtns[j].classList.remove('active');
      self.loadFile(e.target.files[0]);
    }
  });

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
    if (self.particleSystem) self.particleSystem.material.uniforms.uColormap.value = cmap;
    if (self.sliceMesh) self.sliceMesh.material.uniforms.uColormap.value = cmap;
    self.updateColorBar();
  });
  $('ctrl-slice').addEventListener('input', function (e) {
    var pos = parseFloat(e.target.value);
    $('val-slice').textContent = pos.toFixed(2);
    if (self.sliceMesh) self.sliceMesh.material.uniforms.uSlicePos.value = pos;
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

  if (this.particleSystem && this.particleSystem.visible) {
    this.particleSystem.material.uniforms.uSpeed.value =
      parseFloat(document.getElementById('ctrl-pspeed').value);
  }

  this.renderer.render(this.scene, this.camera);
};

// ---- Boot ----
window.app = new WindTunnelApp();

})();
