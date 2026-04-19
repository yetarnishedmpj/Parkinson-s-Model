// --- Configuration ---
const IS_LOCAL_FILE = window.location.protocol === 'file:';
const HOST = IS_LOCAL_FILE ? "localhost:8000" : window.location.host;
const PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

const WS_URL = `${PROTOCOL}//${HOST}/ws`;
const WS_URL_V1 = `${PROTOCOL}//${HOST}/api/v1/ws`;
const API_BASE = `${window.location.protocol}//${HOST}/api/v1`;

// --- Global State ---
let ws;
let scene, camera, renderer, avatar, chart, controls;
let threeContainer, ctx;

// Mall Architecture
let floorGroups = {}; // 0: Ground, 1: Upper
let currentFloor = 0;
let obstacleMeshes = [];

// Humanoid Components
let head, torso, armL, armR, legL, legR;
let npcs = {}; // id -> { mesh, targetPos, curPos }

// Smooth interpolation states
let targetPos = (typeof THREE !== 'undefined') ? new THREE.Vector3(0, 0.7, 0) : { x: 0, y: 0.7, z: 0, set: function(x,y,z){this.x=x;this.y=y;this.z=z;} };
let currentPos = (typeof THREE !== 'undefined') ? new THREE.Vector3(0, 0.7, 0) : { x: 0, y: 0.7, z: 0, lerp: function(t, alpha){this.x += (t.x-this.x)*alpha; this.y += (t.y-this.y)*alpha; this.z += (t.z-this.z)*alpha;} };
let tremorIntensity = 0;
let breathPhase     = 0;
let swayFactor      = 0;
let isFestinating   = false;
let isFreezing      = false;
let impactShake     = 0;

const keys = { w: false, a: false, s: false, d: false };

function setDiagnostic(status, error = "") {
    const overlay = document.getElementById("diag-overlay");
    const statusEl = document.getElementById("diag-status");
    const errorEl = document.getElementById("diag-error");
    if (!overlay) return;
    if (status === "hide") { overlay.classList.add("hidden"); return; }
    overlay.classList.remove("hidden");
    statusEl.innerText = status;
    errorEl.innerText = error;
}

function updateAIInsights(data) {
    const insightText   = document.getElementById("ai-insight-text");
    const insightStatus = document.getElementById("ai-status-long");
    if (data.insight) {
        if (insightText)   insightText.innerText   = data.insight.split(" | ")[0];
        if (insightStatus) insightStatus.innerText = data.insight;
    }
}

function init() {
    requestAnimationFrame(async () => {
        try {
            threeContainer = document.getElementById('three-viewport');
            const canvas = document.getElementById('vitalsChart');
            if (!threeContainer || !canvas) throw new Error("DOM elements missing");
            ctx = canvas.getContext('2d');
            if (typeof THREE === 'undefined') throw new Error("Three.js failed to load.");

            setupThreeJS();
            // Fetch Mall Layout dynamically
            const res = await fetch(`${API_BASE}/mall`);
            const mallData = await res.json();
            buildMallEnvironment(mallData);

            chart = setupCharts();
            connectWebSocket(chart);
            setupKeyboardControls();
            animate();
            window.addEventListener('resize', onWindowResize, false);
        } catch (e) {
            console.error(e);
            setDiagnostic("System Error", e.message);
        }
    });
}

function createAvatarEntity(colorHex, isMain = false) {
    const group = new THREE.Group();
    const mat = new THREE.MeshPhongMaterial({ color: colorHex, emissive: colorHex, emissiveIntensity: 0.5, shininess: 100 });
    
    const _torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.4, 4, 16), mat);
    _torso.position.y = 1.0;
    group.add(_torso);
    
    const _head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), mat);
    _head.position.y = 1.35;
    group.add(_head);
    
    const limbGeo = new THREE.CapsuleGeometry(0.06, 0.4, 4, 8);
    const _armL = new THREE.Mesh(limbGeo, mat);
    _armL.position.set(-0.25, 1.1, 0);
    group.add(_armL);
    
    const _armR = new THREE.Mesh(limbGeo, mat);
    _armR.position.set(0.25, 1.1, 0);
    group.add(_armR);

    const _legL = new THREE.Mesh(limbGeo, mat);
    _legL.position.set(-0.12, 0.6, 0);
    group.add(_legL);

    const _legR = new THREE.Mesh(limbGeo, mat);
    _legR.position.set(0.12, 0.6, 0);
    group.add(_legR);

    if (isMain) {
        torso = _torso; head = _head; armL = _armL; armR = _armR; legL = _legL; legR = _legR;
    }
    
    return group;
}

function setupThreeJS() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070a);
    scene.fog = new THREE.FogExp2(0x05070a, 0.04);

    const w = threeContainer.clientWidth, h = threeContainer.clientHeight;
    camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    threeContainer.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    // Disable OrbitControls keyboard so WASD events reach our handler
    controls.enableKeys = false;
    // Prevent canvas from stealing keyboard focus on click
    renderer.domElement.setAttribute('tabindex', '-1');
    renderer.domElement.style.outline = 'none';
    renderer.domElement.addEventListener('mousedown', (e) => { e.preventDefault(); window.focus(); });

    floorGroups[0] = new THREE.Group();
    floorGroups[1] = new THREE.Group();
    floorGroups[1].position.y = 8.0;
    floorGroups[1].visible = false;
    scene.add(floorGroups[0]);
    scene.add(floorGroups[1]);

    // Build heatmap floor plane for floor 0
    heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = heatmapCanvas.height = 256;
    heatmapCtx = heatmapCanvas.getContext('2d');
    heatmapCtx.fillStyle = 'rgba(0,0,0,0)';
    heatmapCtx.fillRect(0, 0, 256, 256);
    heatmapTexture = new THREE.CanvasTexture(heatmapCanvas);
    const floorGeo = new THREE.PlaneGeometry(30, 30);
    floorMesh = new THREE.Mesh(floorGeo,
        new THREE.MeshBasicMaterial({ map: heatmapTexture, transparent: true, opacity: 0.6, depthWrite: false }));
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.y = 0.02;
    floorMesh.visible = false;
    floorGroups[0].add(floorMesh);

    [0, 1].forEach(f => {
        const grid = new THREE.GridHelper(30, 30, 0x00f2fe, 0x1e293b);
        grid.position.y = 0.01;
        floorGroups[f].add(grid);
    });

    avatar = createAvatarEntity(0x00f2fe, true);
    floorGroups[0].add(avatar);

    scene.add(new THREE.AmbientLight(0xffffff, 0.2));
    const spotlight = new THREE.PointLight(0x00f2fe, 2, 20);
    spotlight.position.set(0, 5, 0);
    scene.add(spotlight);

    camera.position.set(0, 10, 16);
}

// ── Heatmap helpers ────────────────────────────────────────────────────────
function paintHeatmap(freezeList) {
    if (!freezeList || !freezeList.length) return;
    freezeList.forEach(({ x, z, v }) => {
        const px = ((x + 15) / 30) * 256;
        const py = ((z + 15) / 30) * 256;
        const r  = Math.min(30, 8 + v * 6);
        const alpha = Math.min(0.7, 0.15 + v * 0.1);
        const grad = heatmapCtx.createRadialGradient(px, py, 0, px, py, r);
        grad.addColorStop(0, `rgba(255,23,68,${alpha})`);
        grad.addColorStop(1, 'rgba(255,23,68,0)');
        heatmapCtx.fillStyle = grad;
        heatmapCtx.fillRect(px - r, py - r, r * 2, r * 2);
    });
    heatmapTexture.needsUpdate = true;
}

window.toggleHeatmap = function() {
    heatmapVisible = !heatmapVisible;
    if (floorMesh) floorMesh.visible = heatmapVisible;
    const btn = document.getElementById('btn-heatmap');
    if (btn) btn.textContent = heatmapVisible ? 'Heatmap ON' : 'Heatmap OFF';
    if (heatmapVisible) btn.classList.add('active');
    else btn.classList.remove('active');
};

// ── 3D Visual Interventions ────────────────────────────────────────────────
function clearInterventions() {
    interventionMeshes.forEach(m => { if (m.parent) m.parent.remove(m); });
    interventionMeshes = [];
}

function renderInterventionCues(actionIdx, pos) {
    clearInterventions();
    const fg = floorGroups[currentFloor];

    if (actionIdx === 1) {
        // RAS: pulsing cyan rings around avatar
        for (let i = 0; i < 3; i++) {
            const ring = new THREE.Mesh(
                new THREE.TorusGeometry(0.5 + i * 0.4, 0.04, 8, 32),
                new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.8 - i * 0.2 })
            );
            ring.rotation.x = Math.PI / 2;
            ring.position.set(pos.x, 0.05, pos.z);
            fg.add(ring); interventionMeshes.push(ring);
        }
    } else if (actionIdx === 2) {
        // Visual Laser: red line projected forward
        const pts = [new THREE.Vector3(pos.x, 0.05, pos.z),
                     new THREE.Vector3(pos.x, 0.05, pos.z - 3)];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff1744, linewidth: 3 }));
        fg.add(line); interventionMeshes.push(line);
    } else if (actionIdx === 3) {
        // DBS flash: gold emissive glow on avatar
        if (torso) { torso.material.emissive.setHex(0xffaa00); torso.material.emissiveIntensity = 1.2; }
        setTimeout(() => { if (torso) { torso.material.emissive.setHex(0x00f2fe); torso.material.emissiveIntensity = 0.5; }}, 600);
    }
    if (actionIdx === 1 || actionIdx === 2) setTimeout(clearInterventions, 2000);
}

// ── Session Replay ─────────────────────────────────────────────────────────
window.toggleReplayPanel = async function() {
    replayPanelOpen = !replayPanelOpen;
    const panel = document.getElementById('replay-panel');
    if (!panel) return;
    panel.classList.toggle('hidden', !replayPanelOpen);
    if (!replayPanelOpen) return;
    const list = document.getElementById('session-list');
    list.innerHTML = 'Loading…';
    try {
        const res = await fetch(`${API_BASE}/sessions`);
        const sessions = await res.json();
        if (!sessions.length) { list.innerHTML = 'No sessions recorded yet.'; return; }
        list.innerHTML = sessions.map(s =>
            `<div class="replay-session-item" onclick="loadReplay('${s.id}')">
                <b>${s.started_at.slice(0,19).replace('T',' ')}</b><br>
                ${s.tick_count} ticks &nbsp;${s.ended_at ? '✓' : '⚡ live'}
             </div>`
        ).join('');
    } catch(e) { list.innerHTML = 'Error loading sessions.'; }
};

window.loadReplay = async function(sessionId) {
    stopReplay();
    const res   = await fetch(`${API_BASE}/sessions/${sessionId}/replay`);
    const data  = await res.json();
    replayTicks = data.ticks;
    replayIndex = 0;
    document.getElementById('vp-label-text').textContent = 'REPLAY MODE';
    replayTimer = setInterval(() => {
        if (replayIndex >= replayTicks.length) { stopReplay(); return; }
        const tick = replayTicks[replayIndex++];
        const v = tick.vitals;
        if (v && v.position) targetPos.set(v.position.x, 0.7, v.position.z);
        if (v) { tremorIntensity = v.tremor_intensity || 0; isFreezing = v.is_freezing || false; }
    }, 50);
};

window.stopReplay = function() {
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
    replayTicks = []; replayIndex = 0;
    const lbl = document.getElementById('vp-label-text');
    if (lbl) lbl.textContent = 'Spatial Navigation Layer';
};

function buildMallEnvironment(mallData) {
    const obstacleColors = [0x2ed573, 0xff4757, 0xffa502, 0xa29bfe, 0x55efc4];
    
    mallData.obstacles.forEach((obs, idx) => {
        const color = obstacleColors[idx % obstacleColors.length];
        const group = floorGroups[obs.floor];
        let geo, edgesGeo, shapeClass;

        if (obs.type === "circle") {
            geo = new THREE.CylinderGeometry(obs.radius, obs.radius, 2.5, 20);
            edgesGeo = geo;
            shapeClass = 'circle';
        } else if (obs.type === "rect") {
            geo = new THREE.BoxGeometry(obs.w, 2.5, obs.d);
            edgesGeo = geo;
            shapeClass = 'rect';
        }

        const body = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
            color: color, transparent: true, opacity: 0.3,
            emissive: color, emissiveIntensity: 0.2
        }));
        body.position.set(obs.x, 1.25, obs.z);
        group.add(body);

        const edges = new THREE.LineSegments(
            new THREE.EdgesGeometry(edgesGeo),
            new THREE.LineBasicMaterial({ color: color })
        );
        edges.position.copy(body.position);
        group.add(edges);

        obstacleMeshes.push({ body, obj: obs });
    });

    // Escalators (Visual connecting beams)
    mallData.escalators.forEach((esc) => {
        const beam = new THREE.Mesh(
            new THREE.CylinderGeometry(1.0, 1.0, 12, 16),
            new THREE.MeshBasicMaterial({color: 0xffff00, wireframe: true, transparent:true, opacity:0.3})
        );
        beam.position.set(esc.x, 4.0, esc.z);
        scene.add(beam); // belongs to global scene
    });
}

function setupCharts() {
    return new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ borderColor: '#00f2fe', backgroundColor: 'rgba(0, 242, 254, 0.1)', data: [], borderWidth: 2, fill: true }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { min: 40, max: 180 } }, elements: { point: { radius: 0 } }, plugins: { legend: { display: false } }, animation: { duration: 0 } }
    });
}

function connectWebSocket(chart) {
    ws = new WebSocket(WS_URL);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        const v = msg.vitals || {};
        const a = msg.analytics || {};
        
        setDiagnostic("hide");
        updateAIInsights(a);

        // Handle Floor Changes
        if (v.position.floor !== currentFloor) {
            // Remove avatar from old, add to new
            floorGroups[currentFloor].remove(avatar);
            currentFloor = v.position.floor;
            floorGroups[currentFloor].add(avatar);
            
            // Switch rendering visibility
            floorGroups[0].visible = (currentFloor === 0);
            floorGroups[1].visible = (currentFloor === 1);
            
            // Move camera
            const yOffset = currentFloor === 1 ? 8.0 : 0.0;
            camera.position.y = 10 + yOffset;
            controls.target.set(0, 1 + yOffset, 0);
            
            // UI Update
            const vpLabel = document.querySelector('.vp-label-text');
            if (vpLabel) vpLabel.innerText = `Mall Environment — Floor ${currentFloor}`;
        }

        targetPos.set(v.position.x, 0.7, v.position.z);
        tremorIntensity  = v.tremor_intensity   || 0;
        breathPhase      = v.breath_phase        || 0;
        swayFactor       = v.sway                || 0;
        isFestinating    = v.festination_active  || false;
        isFreezing       = v.is_freezing         || false;

        // Process NPCs
        if (v.npcs) {
            v.npcs.forEach(n => {
                if (!npcs[n.id]) {
                    const m = createAvatarEntity(0xff4757, false); // Red generic pedestrians
                    floorGroups[n.floor].add(m);
                    npcs[n.id] = { mesh: m, targetPos: new THREE.Vector3(n.x, 0.7, n.z), curPos: new THREE.Vector3(n.x, 0.7, n.z), floor: n.floor };
                } else {
                    // Update existing
                    if (npcs[n.id].floor !== n.floor) {
                        floorGroups[npcs[n.id].floor].remove(npcs[n.id].mesh);
                        npcs[n.id].floor = n.floor;
                        floorGroups[n.floor].add(npcs[n.id].mesh);
                    }
                    npcs[n.id].targetPos.set(n.x, 0.7, n.z);
                }
            });
        }

        // Goal indicator
        const goalEl = document.getElementById('goal-name');
        if (goalEl && v.current_goal) goalEl.textContent = v.current_goal;

        // LSTM pill
        const lstmPill = document.getElementById('lstm-pill');
        if (lstmPill) lstmPill.classList.toggle('active', !!(a.using_lstm));

        // FOG risk bar
        const fogFill = document.getElementById('freeze-prob-fill');
        if (fogFill) fogFill.style.width = `${Math.round((a.freeze_prob_3s||0)*100)}%`;

        // Heatmap paint
        if (v.freeze_heatmap && v.freeze_heatmap.length) paintHeatmap(v.freeze_heatmap);

        // 3D Intervention cues (fire when action != 0)
        if (a.rl_action && a.rl_action !== 0) renderInterventionCues(a.rl_action, v.position);

        // Tiles
        const setTile = (tId, vId, bId, d, pct, s) => {
            const el=document.getElementById(vId), bar=document.getElementById(bId), t=document.getElementById(tId);
            if(el) el.innerText=d; if(bar) bar.style.width=`${Math.min(100,Math.max(0,pct))}%`;
            if(t) t.className=`vital-tile ${s||''}`;
        };
        const hr=Math.round(v.heart_rate), stress=Math.round((a.stress_level||0)*100);
        const fall=Math.round((a.fall_risk||0)*100), fatigue=Math.round((v.fatigue||0)*100);
        setTile('tile-hr','val-hr','bar-hr', hr, (hr-40)/140*100, hr>130?'state-crit':hr>110?'state-warn':'state-ok');
        setTile('tile-stress','val-stress','bar-stress', `${stress}%`, stress, stress>70?'state-crit':stress>40?'state-warn':'state-ok');
        setTile('tile-fall','val-fall','bar-fall', `${fall}%`, fall, fall>60?'state-crit':fall>30?'state-warn':'state-ok');
        setTile('tile-fatigue','val-fatigue','bar-fatigue', `${fatigue}%`, fatigue, fatigue>70?'state-crit':fatigue>40?'state-warn':'state-ok');

        const badge = document.getElementById('ai-status-badge');
        if (badge) { const s=a.status||'NOMINAL'; badge.innerText=`● ${s}`; badge.className=`ai-status-badge ${s.toLowerCase()}`; }

        const fBadge = document.getElementById('freeze-badge');
        if (fBadge) fBadge.className = 'alert-badge' + (isFreezing ? ' freeze' : '');

        chart.data.labels.push('');
        chart.data.datasets[0].data.push(v.heart_rate);
        if (chart.data.labels.length > 20) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
        chart.update('none');
    };

    ws.onclose = () => { setTimeout(() => connectWebSocket(chart), 3000); };
}

function setupKeyboardControls() {
    // Track key state
    window.addEventListener('keydown', (e) => {
        const k = e.key.toLowerCase();
        if (keys.hasOwnProperty(k)) {
            e.preventDefault(); // prevent page scroll on Space/arrows
            keys[k] = true;
        }
    });
    window.addEventListener('keyup', (e) => {
        const k = e.key.toLowerCase();
        if (keys.hasOwnProperty(k)) keys[k] = false;
    });

    // Continuously send control vector at ~20 Hz while any key is held
    // This keeps backend is_manual alive and prevents missed events
    setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const anyKey = keys.w || keys.a || keys.s || keys.d;
        let x = 0, z = 0;
        if (keys.w) z -= 1;
        if (keys.s) z += 1;
        if (keys.a) x -= 1;
        if (keys.d) x += 1;
        // Always send to reset is_manual=False when keys released
        ws.send(JSON.stringify({ type: "control", x, z }));

        // Visual HUD indicator
        const hud = document.getElementById('wasd-hud');
        if (hud) {
            hud.querySelector('#k-w').classList.toggle('active', keys.w);
            hud.querySelector('#k-a').classList.toggle('active', keys.a);
            hud.querySelector('#k-s').classList.toggle('active', keys.s);
            hud.querySelector('#k-d').classList.toggle('active', keys.d);
        }
    }, 50);
}

function onWindowResize() {
    camera.aspect = threeContainer.clientWidth / threeContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
}

function handleFrontendCollision() {
    const AVATAR_R = 0.3;
    const CAGE_H   = 14.7;

    obstacleMeshes.forEach((meshObj) => {
        const obs = meshObj.obj;
        if (obs.floor !== currentFloor) return;
        
        let dx, dz, dist, influence;
        
        if (obs.type === "circle") {
            dx = currentPos.x - obs.x;
            dz = currentPos.z - obs.z;
            dist = Math.sqrt(dx*dx + dz*dz);
            influence = obs.radius + 3.0;
        } else if (obs.type === "rect") {
            dx = Math.max(Math.abs(currentPos.x - obs.x) - obs.w/2, 0);
            dz = Math.max(Math.abs(currentPos.z - obs.z) - obs.d/2, 0);
            dist = Math.sqrt(dx*dx + dz*dz);
            influence = 3.0;
        }

        if (dist === 0) return;
        const prox = Math.max(0, 1.0 - dist / influence);

        // Proximity glow
        meshObj.body.material.emissiveIntensity = 0.15 + prox * 1.05;
        meshObj.body.material.opacity = 0.3 + prox * 0.4;
    });

    const wallHit = (currentPos.x < -CAGE_H || currentPos.x > CAGE_H || currentPos.z < -CAGE_H || currentPos.z > CAGE_H);
    if (wallHit) {
        currentPos.x = Math.max(-CAGE_H, Math.min(CAGE_H, currentPos.x));
        currentPos.z = Math.max(-CAGE_H, Math.min(CAGE_H, currentPos.z));
    }
}

function animateNPCs() {
    Object.values(npcs).forEach(npc => {
        npc.curPos.lerp(npc.targetPos, 0.1);
        npc.mesh.position.copy(npc.curPos);
        
        const velX = npc.targetPos.x - npc.curPos.x;
        const velZ = npc.targetPos.z - npc.curPos.z;
        const speed = Math.sqrt(velX*velX + velZ*velZ);
        
        if (speed > 0.001) {
            const tgtRot = Math.atan2(velX, velZ);
            let dRot = tgtRot - npc.mesh.rotation.y;
            while (dRot > Math.PI) dRot -= 2*Math.PI; while (dRot < -Math.PI) dRot += 2*Math.PI;
            npc.mesh.rotation.y += Math.sign(dRot) * Math.min(Math.abs(dRot), 0.15);
        }
    });
}

function animate() {
    requestAnimationFrame(animate);
    
    currentPos.lerp(targetPos, 0.1);
    handleFrontendCollision();
    impactShake *= 0.82;
    
    if (avatar) {
        avatar.position.copy(currentPos);
        animateNPCs();

        const time  = Date.now() * 0.001;
        const velX  = targetPos.x - currentPos.x;
        const velZ  = targetPos.z - currentPos.z;
        const speed = Math.sqrt(velX**2 + velZ**2);
        const normSpeed = Math.min(speed * 80, 1.0);

        const tremorFactor  = Math.max(0, tremorIntensity * (1.0 - normSpeed * 0.5));
        const tremorX       = Math.sin(time * 28.0) * tremorFactor * 0.06;
        const tremorZ       = Math.sin(time * 34.0) * tremorFactor * 0.04;
        const stoop         = isFreezing ? 0.45 : (0.28 + normSpeed * 0.08);

        torso.rotation.x = stoop;
        head.rotation.x = 0.3 + (isFreezing ? 0.2 : 0);
        head.rotation.z = tremorX * 2.0;
        
        if (speed > 0.0005) {
            const tgtY = Math.atan2(velX, velZ);
            let dy = tgtY - avatar.rotation.y;
            while (dy > Math.PI) dy -= 2*Math.PI; while (dy < -Math.PI) dy += 2*Math.PI;
            avatar.rotation.y += Math.sign(dy) * Math.min(Math.abs(dy), 0.08);
        }

        if (isFreezing) {
            legL.rotation.x = 0.08 + tremorX; legR.rotation.x = -0.08 + tremorZ;
            armL.rotation.x = -0.3 + tremorX; armR.rotation.x = -0.3 + tremorZ;
            avatar.rotation.z = tremorX * 0.4;
            avatar.rotation.x = stoop + Math.abs(tremorZ) * 0.15;
            avatar.position.y = 0.1;
        } else {
            const shufFreq = time * 18 * Math.max(normSpeed, 0.1);
            const shufAmp = normSpeed * 0.12;
            legL.rotation.x = Math.sin(shufFreq) * shufAmp + tremorX;
            legR.rotation.x = Math.sin(shufFreq + Math.PI) * shufAmp + tremorZ;
            armL.rotation.x = -0.1 + Math.sin(shufFreq) * 0.04;
            armR.rotation.x = -0.1 + Math.sin(shufFreq + Math.PI) * 0.04;
            avatar.rotation.z = swayFactor * 0.04;
            avatar.rotation.x = stoop;
            avatar.position.y = 0.1 + Math.abs(Math.sin(shufFreq) * 0.015 * normSpeed);
        }
    }

    if (controls) controls.update();
    renderer.render(scene, camera);
}

window.setScenario = function(name) {
    fetch(`${API_BASE}/scenario/${name}`, { method: "POST" });
    document.querySelectorAll('.sc-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`btn-${name}`).classList.add('active');
};

init();
