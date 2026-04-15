
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

// Humanoid Components
let head, torso, armL, armR;

// Smooth interpolation states
let targetPos = (typeof THREE !== 'undefined') ? new THREE.Vector3(0, 0.7, 0) : { x: 0, y: 0.7, z: 0, set: function(x,y,z){this.x=x;this.y=y;this.z=z;} };
let currentPos = (typeof THREE !== 'undefined') ? new THREE.Vector3(0, 0.7, 0) : { x: 0, y: 0.7, z: 0, lerp: function(t, alpha){this.x += (t.x-this.x)*alpha; this.y += (t.y-this.y)*alpha; this.z += (t.z-this.z)*alpha;} };
let targetRotY = 0;
let currentRotY = 0;
let tremorIntensity = 0;
let breathPhase     = 0;
let swayFactor      = 0;
let isFestinating   = false;  // backend-driven festination state
let isFreezing      = false;  // freeze-of-gait state
let obstacleMeshes  = [];     // { body } refs for per-obstacle effects
let impactShake     = 0;      // decays after collision — shakes avatar body

const obstacles = [
    { name: "Doorway",      x:  0.0, z: -8.0, radius: 1.8, color: 0x2ed573 },
    { name: "Stairs",       x:  8.0, z:  8.0, radius: 2.0, color: 0xff4757 },
    { name: "Narrow Hall",  x: -8.0, z:  2.0, radius: 1.8, color: 0xffa502 },
    { name: "Pillar",       x:  4.0, z: -3.0, radius: 1.2, color: 0xff6b81 },
    { name: "Furniture",    x: -5.0, z: -6.0, radius: 1.5, color: 0xeccc68 },
    { name: "Wall Block",   x:  7.0, z:  0.0, radius: 1.2, color: 0xa29bfe },
    { name: "Table",        x: -2.0, z:  7.0, radius: 1.3, color: 0x55efc4 },
    { name: "Cabinet",      x:  2.0, z: 11.0, radius: 1.3, color: 0xfd79a8 },
];

const keys = { w: false, a: false, s: false, d: false };

// --- UI / Diagnostic Helpers ---
function setDiagnostic(status, error = "") {
    const overlay = document.getElementById("diag-overlay");
    const statusEl = document.getElementById("diag-status");
    const errorEl = document.getElementById("diag-error");
    if (!overlay) return;
    
    if (status === "hide") {
        overlay.classList.add("hidden");
        return;
    }
    
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

// --- Initialization ---
function init() {
    // Defer to after first paint so CSS layout is computed and
    // threeContainer.clientWidth/clientHeight are non-zero.
    requestAnimationFrame(() => {
        try {
            threeContainer = document.getElementById('three-viewport');
            const canvas = document.getElementById('vitalsChart');
            if (!threeContainer || !canvas) throw new Error("DOM elements missing");
            ctx = canvas.getContext('2d');

            if (typeof THREE === 'undefined') throw new Error("Three.js failed to load.");

            setupThreeJS();
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

function setupThreeJS() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070a);
    scene.fog = new THREE.FogExp2(0x05070a, 0.05);

    const w = threeContainer.clientWidth  || window.innerWidth  * 0.65;
    const h = threeContainer.clientHeight || window.innerHeight - 92;

    camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    threeContainer.appendChild(renderer.domElement);
    // Force correct size now that canvas is in the DOM
    setTimeout(onWindowResize, 0);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 1, 0);

    // Environment — Grid (clipped to cage footprint)
    const grid = new THREE.GridHelper(26, 26, 0x00f2fe, 0x1e293b);
    grid.position.y = 0.01;
    scene.add(grid);

    // Cage wireframe (hard physics boundary = ±13 units)
    const CAGE = 26;
    const cageEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(CAGE, 8, CAGE));
    const cage = new THREE.LineSegments(
        cageEdges,
        new THREE.LineBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.55 })
    );
    cage.position.y = 4;
    scene.add(cage);

    // Semi-transparent cage wall panels (subtle glow)
    const wallMat = new THREE.MeshBasicMaterial({
        color: 0x00f2fe, transparent: true, opacity: 0.04, side: THREE.DoubleSide
    });
    [[-1,'x'],[1,'x'],[-1,'z'],[1,'z']].forEach(([side, axis]) => {
        const w = new THREE.Mesh(new THREE.PlaneGeometry(CAGE, 8), wallMat);
        w.position.y = 4;
        if (axis === 'x') { w.rotation.y = Math.PI / 2; w.position.x = side * CAGE / 2; }
        else              { w.position.z = side * CAGE / 2; }
        scene.add(w);
    });

    // Humanoid Procedural Model (Phase 4 articulated)
    avatar = new THREE.Group();
    
    const mat = new THREE.MeshPhongMaterial({ color: 0x00f2fe, emissive: 0x00f2fe, emissiveIntensity: 0.5, shininess: 100 });
    
    // Torso & Head
    torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.4, 4, 16), mat);
    torso.position.y = 1.0;
    avatar.add(torso);
    
    head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), mat);
    head.position.y = 1.35;
    avatar.add(head);
    
    // Arms
    const limbGeo = new THREE.CapsuleGeometry(0.06, 0.4, 4, 8);
    armL = new THREE.Mesh(limbGeo, mat);
    armL.position.set(-0.25, 1.1, 0);
    avatar.add(armL);
    
    armR = new THREE.Mesh(limbGeo, mat);
    armR.position.set(0.25, 1.1, 0);
    avatar.add(armR);

    // Legs (New for Phase 4)
    legL = new THREE.Mesh(limbGeo, mat);
    legL.position.set(-0.12, 0.6, 0);
    avatar.add(legL);

    legR = new THREE.Mesh(limbGeo, mat);
    legR.position.set(0.12, 0.6, 0);
    avatar.add(legR);

    scene.add(avatar);

    obstacles.forEach(obs => {
        // Cylinder height + radius matches backend circle-collision model
        const body = new THREE.Mesh(
            new THREE.CylinderGeometry(obs.radius, obs.radius, 2.5, 20),
            new THREE.MeshPhongMaterial({
                color: obs.color, transparent: true, opacity: 0.35,
                emissive: obs.color, emissiveIntensity: 0.2
            })
        );
        body.position.set(obs.x, 1.25, obs.z);
        scene.add(body);

        // Bright edge wireframe to reinforce solidity
        const edges = new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.CylinderGeometry(obs.radius, obs.radius, 2.5, 20)),
            new THREE.LineBasicMaterial({ color: obs.color })
        );
        edges.position.copy(body.position);
        scene.add(edges);

        // Floor ring — shows exact collision boundary on the ground plane
        const ring = new THREE.LineLoop(
            new THREE.CircleGeometry(obs.radius, 32),
            new THREE.LineBasicMaterial({ color: obs.color, transparent: true, opacity: 0.6 })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(obs.x, 0.02, obs.z);
        scene.add(ring);

        obstacleMeshes.push({ body, ring });   // store for per-frame effects
    });

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.2));
    const spotlight = new THREE.PointLight(0x00f2fe, 2, 20);
    spotlight.position.set(0, 5, 0);
    scene.add(spotlight);

    camera.position.set(0, 16, 22);
}

function setupCharts() {
    const chartData = {
        labels: [],
        datasets: [{
            label: 'Heart Rate',
            borderColor: '#00f2fe',
            backgroundColor: 'rgba(0, 242, 254, 0.1)',
            data: [],
            borderWidth: 2,
            tension: 0.4,
            fill: true
        }]
    };

    return new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: false },
                y: {
                    min: 40,
                    max: 180,
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b', font: { size: 10 } }
                }
            },
            elements: {
                point: { radius: 0 }   // clean ECG line, no dots
            },
            plugins: { legend: { display: false } },
            animation: { duration: 0 }  // no animation lag at 20 Hz
        }
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

        // Interpolation Targets
        targetPos.set(v.position.x, 0.7, v.position.z);
        tremorIntensity  = v.tremor_intensity   || 0;
        breathPhase      = v.breath_phase        || 0;
        swayFactor       = v.sway                || 0;
        isFestinating    = v.festination_active  || false;
        isFreezing       = v.is_freezing         || false;

        // ── Helper: set tile value + bar + color state ──
        const setTile = (tileId, valId, barId, displayVal, pct, state) => {
            const tile = document.getElementById(tileId);
            const el   = document.getElementById(valId);
            const bar  = document.getElementById(barId);
            if (el)  el.innerText = displayVal;
            if (bar) bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
            if (tile) { tile.className = 'vital-tile ' + (state || ''); }
        };

        const hr     = Math.round(v.heart_rate);
        const stress = Math.round((a.stress_level || 0) * 100);
        const fall   = Math.round((a.fall_risk   || 0) * 100);
        const health = +(a.health_index || 0);
        const trem   = +((v.tremor_intensity || 0) * 100).toFixed(0);
        const act    = +((v.activity_level  || 0) * 10).toFixed(1);

        // Heart Rate: normal 60-100, warn >110, crit >130
        const hrState = hr > 130 ? 'state-crit' : hr > 110 ? 'state-warn' : 'state-ok';
        setTile('tile-hr', 'val-hr', 'bar-hr', hr, (hr - 40) / 140 * 100, hrState);

        // Stress
        const stressState = stress > 70 ? 'state-crit' : stress > 40 ? 'state-warn' : 'state-ok';
        setTile('tile-stress', 'val-stress', 'bar-stress', stress + '%', stress, stressState);

        // Fall Risk
        const fallState = fall > 60 ? 'state-crit' : fall > 30 ? 'state-warn' : 'state-ok';
        setTile('tile-fall', 'val-fall', 'bar-fall', fall + '%', fall, fallState);

        // Tremor
        const tremState = trem > 60 ? 'state-crit' : trem > 30 ? 'state-warn' : 'state-ok';
        setTile('tile-tremor', 'val-tremor', 'bar-tremor', trem, trem, tremState);

        // Health Index
        const healthState = health < 70 ? 'state-crit' : health < 85 ? 'state-warn' : 'state-ok';
        setTile('tile-health', 'val-health', 'bar-health', health.toFixed(0), health, healthState);
        const hs = document.getElementById('health-status');
        if (hs) { hs.innerText = a.status || 'NOMINAL'; hs.className = healthState.replace('state-','status-'); }

        // Activity
        setTile('tile-activity', 'val-activity', 'bar-activity', act, act * 10, 'state-ok');

        // AI diagnosis badge
        const badge = document.getElementById('ai-status-badge');
        if (badge) {
            const s = a.status || 'NOMINAL';
            badge.innerText = '● ' + s;
            badge.style.background = s.includes('CRITICAL') || s === 'FREEZE' ? 'rgba(255,23,68,0.2)'
                                   : s === 'ELEVATED' ? 'rgba(255,171,0,0.15)' : 'rgba(0,229,255,0.1)';
            badge.style.color      = s.includes('CRITICAL') || s === 'FREEZE' ? 'var(--red)'
                                   : s === 'ELEVATED' ? 'var(--amber)' : 'var(--cyan)';
            badge.style.borderColor= s.includes('CRITICAL') || s === 'FREEZE' ? 'rgba(255,23,68,0.4)'
                                   : s === 'ELEVATED' ? 'rgba(255,171,0,0.4)' : 'var(--border-accent)';
        }

        // Freeze / festination topbar alerts
        const freezeBadge = document.getElementById('freeze-badge');
        const festinBadge = document.getElementById('festin-badge');
        if (freezeBadge) { freezeBadge.className = 'alert-badge' + (isFreezing     ? ' freeze' : ''); }
        if (festinBadge) { festinBadge.className = 'alert-badge' + (isFestinating  ? ' festin' : ''); }

        // Chart Sync
        chart.data.labels.push('');
        chart.data.datasets[0].data.push(v.heart_rate);
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update('none');
    };

    ws.onclose = () => { setTimeout(() => connectWebSocket(chart), 3000); };
}

function setupKeyboardControls() {
    window.addEventListener('keydown', (e) => { if (keys.hasOwnProperty(e.key.toLowerCase())) { keys[e.key.toLowerCase()] = true; sendControlUpdate(); } });
    window.addEventListener('keyup', (e) => { if (keys.hasOwnProperty(e.key.toLowerCase())) { keys[e.key.toLowerCase()] = false; sendControlUpdate(); } });

    function sendControlUpdate() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        let x = 0, z = 0;
        if (keys.w) z -= 1; if (keys.s) z += 1;
        if (keys.a) x -= 1; if (keys.d) x += 1;
        ws.send(JSON.stringify({ type: "control", x, z }));
    }
}

function onWindowResize() {
    camera.aspect = threeContainer.clientWidth / threeContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
}

// ─────────────────────────────────────────────────────────────────────────────
// FRONTEND COLLISION  —  runs every frame, independent of backend physics
// Prevents avatar from visually entering any obstacle or the cage walls.
// ─────────────────────────────────────────────────────────────────────────────
function handleFrontendCollision() {
    const AVATAR_R = 0.3;
    const CAGE_H   = 12.7;  // 13.0 cage_half minus avatar radius

    obstacles.forEach((obs, idx) => {
        const dx   = currentPos.x - obs.x;
        const dz   = currentPos.z - obs.z;
        const dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 0.001) return;

        const minDist   = obs.radius + AVATAR_R;
        const influence = obs.radius + 4.0;              // glow starts 4 units away
        const prox      = Math.max(0, 1.0 - dist / influence);

        // ── Proximity glow (scales from 0.15 to 1.2 toward surface) ──
        if (obstacleMeshes[idx]) {
            obstacleMeshes[idx].body.material.emissiveIntensity = 0.15 + prox * 1.05;
            obstacleMeshes[idx].body.material.opacity           = 0.35 + prox * 0.4;
        }

        // ── Hard collision: push avatar out + impact effects ──
        if (dist < minDist) {
            const nx      = dx / dist;
            const nz      = dz / dist;
            const pushOut = minDist - dist;

            // Correct visual position immediately
            currentPos.x += nx * pushOut;
            currentPos.z += nz * pushOut;

            // Nudge the target too so the lerp doesn't keep pulling avatar back in
            targetPos.x  += nx * pushOut * 0.5;
            targetPos.z  += nz * pushOut * 0.5;

            // Flash the obstacle bright white-hot
            if (obstacleMeshes[idx]) {
                obstacleMeshes[idx].body.material.emissiveIntensity = 2.5;
                obstacleMeshes[idx].body.material.opacity           = 0.9;
            }

            // Impact feedback: body shake + tremor spike
            impactShake    = Math.max(impactShake, 0.18);
            tremorIntensity = Math.min(3.0, tremorIntensity + 0.6);
        }
    });

    // ── Cage wall collision ──
    const wallHit = (
        currentPos.x < -CAGE_H || currentPos.x > CAGE_H ||
        currentPos.z < -CAGE_H || currentPos.z > CAGE_H
    );
    if (wallHit) {
        currentPos.x = Math.max(-CAGE_H, Math.min(CAGE_H, currentPos.x));
        currentPos.z = Math.max(-CAGE_H, Math.min(CAGE_H, currentPos.z));
        targetPos.x  = Math.max(-CAGE_H, Math.min(CAGE_H, targetPos.x));
        targetPos.z  = Math.max(-CAGE_H, Math.min(CAGE_H, targetPos.z));
        impactShake  = Math.max(impactShake, 0.12);
    }
}

function animate() {
    requestAnimationFrame(animate);
    
    // Smooth Pos Interpolation
    currentPos.lerp(targetPos, 0.1);

    // ── Collision: enforce EVERY frame before rendering ──
    handleFrontendCollision();
    // Decay shake
    impactShake *= 0.82;
    
    if (avatar) {
        avatar.position.copy(currentPos);

        const time  = Date.now() * 0.001;
        const velX  = targetPos.x - currentPos.x;
        const velZ  = targetPos.z - currentPos.z;
        const speed = Math.sqrt(velX**2 + velZ**2);

        // ── speeds normalised for animation (small real-world values → bigger visual)
        const normSpeed = Math.min(speed * 80, 1.0);   // 0–1 animation scale

        // ────────────────────────────────────────────────────────────────────
        // 1. RESTING TREMOR  (4–6 Hz, worst when still — clinically correct PD)
        //    Reduces during intentional movement, spikes during festination/impact
        // ────────────────────────────────────────────────────────────────────
        const tremorFactor  = Math.max(0, tremorIntensity * (1.0 - normSpeed * 0.5));
        const tremHz4       = Math.sin(time * 28.0);   // ~4.5 Hz resting tremor
        const tremHz5       = Math.sin(time * 34.0);   // ~5.4 Hz hand tremor
        const tremorX       = tremHz4 * tremorFactor * 0.06;
        const tremorZ       = tremHz5 * tremorFactor * 0.04;

        // ────────────────────────────────────────────────────────────────────
        // 2. STOOPED POSTURE  (constant forward lean, more pronounced when frozen)
        // ────────────────────────────────────────────────────────────────────
        const stoop         = isFreezing ? 0.45 : (0.28 + normSpeed * 0.08);
        torso.rotation.x    = stoop;
        head.rotation.x     = 0.3 + (isFreezing ? 0.2 : 0);   // chin down
        head.rotation.z     = tremHz4 * tremorFactor * 0.12;  // pill-rolling head tremor

        // Chest barely expands (PD patients have reduced respiratory amplitude)
        torso.scale.x       = 1.0 + breathPhase * 0.02;
        torso.scale.z       = 1.0 + breathPhase * 0.015;

        // ────────────────────────────────────────────────────────────────────
        // 3. FACING DIRECTION  (jerky cogwheel-style rotation, not smooth slerp)
        // ────────────────────────────────────────────────────────────────────
        if (speed > 0.0005) {
            const targetRotY    = Math.atan2(velX, velZ);
            let   deltaRot      = targetRotY - avatar.rotation.y;
            // Wrap to [-PI, PI]
            while (deltaRot >  Math.PI) deltaRot -= 2 * Math.PI;
            while (deltaRot < -Math.PI) deltaRot += 2 * Math.PI;
            // Cogwheel: snap in discrete ~15° steps rather than smooth glide
            const snapStep      = Math.sign(deltaRot) * Math.min(Math.abs(deltaRot), 0.08);
            avatar.rotation.y  += snapStep;
        }

        // ────────────────────────────────────────────────────────────────────
        // 4. FREEZE-OF-GAIT POSE  (rigid, legs locked, intense tremor)
        // ────────────────────────────────────────────────────────────────────
        if (isFreezing) {
            // Legs completely locked
            legL.rotation.x    =  0.08 + tremorX * 0.5;
            legR.rotation.x    = -0.08 + tremorZ * 0.5;
            // Arms slightly out for balance (reaching)
            armL.rotation.x    = -0.3 + tremorX;
            armR.rotation.x    = -0.3 + tremorZ;
            armL.rotation.z    =  0.25 + tremHz4 * tremorFactor * 0.15;
            armR.rotation.z    = -0.25 + tremHz5 * tremorFactor * 0.15;
            // Full-body shudder
            avatar.rotation.z  = tremorX * 0.4;
            avatar.rotation.x  = stoop + Math.abs(tremorZ) * 0.15;
            head.position.y    = 1.35;
            avatar.position.y  = 0.1;

        // ────────────────────────────────────────────────────────────────────
        // 5. FESTINATION  (frantic tiny shuffling steps, leaning too far forward)
        // ────────────────────────────────────────────────────────────────────
        } else if (isFestinating) {
            const festFreq     = time * 25;   // very rapid cadence
            // Rapid low-amplitude leg shuffle
            legL.rotation.x   = Math.sin(festFreq)       * 0.18;
            legR.rotation.x   = Math.sin(festFreq + Math.PI) * 0.18;
            // Arms barely move (reduced arm swing is a PD hallmark)
            armL.rotation.x   = Math.sin(festFreq)       * 0.04 + tremorX * 0.5;
            armR.rotation.x   = Math.sin(festFreq + Math.PI) * 0.04 + tremorZ * 0.5;
            // Extra forward lean — can't stop, lurching forward
            avatar.rotation.x = stoop + 0.18;
            avatar.rotation.z = tremorX * 0.15;
            const festBob      = Math.abs(Math.sin(festFreq)) * 0.03;
            head.position.y   = 1.35 + festBob;
            avatar.position.y = 0.1 + festBob;

        // ────────────────────────────────────────────────────────────────────
        // 6. NORMAL WALK  (Parkinson's shuffling gait)
        // ────────────────────────────────────────────────────────────────────
        } else {
            // Shuffling: rapid but TINY steps (high freq, low amplitude)
            const shuffleFreq  = time * 18 * Math.max(normSpeed, 0.1);
            const shuffleAmp   = normSpeed * 0.12;          // tiny range
            const bob          = Math.sin(shuffleFreq) * 0.015 * normSpeed;  // almost no vertical bob

            legL.rotation.x   = Math.sin(shuffleFreq)          * shuffleAmp + tremorX * 0.3;
            legR.rotation.x   = Math.sin(shuffleFreq + Math.PI) * shuffleAmp + tremorZ * 0.3;

            // Arms almost stationary — PD patients lose arm swing
            armL.rotation.x   = -0.1 + Math.sin(shuffleFreq) * 0.04 * normSpeed + tremorX;
            armR.rotation.x   = -0.1 + Math.sin(shuffleFreq + Math.PI) * 0.04 * normSpeed + tremorZ;
            // Slight in-rotation (PD arms hang inward)
            armL.rotation.z   =  0.12 + tremorX * 0.2;
            armR.rotation.z   = -0.12 + tremorZ * 0.2;

            avatar.rotation.z = tremorX * 0.08 + swayFactor * 0.04;
            avatar.rotation.x = stoop;
            head.position.y   = 1.35 + bob;
            avatar.position.y = 0.1 + Math.abs(bob);
        }

        // ── Impact body shake (applied on top of everything) ──
        if (impactShake > 0.002) {
            avatar.position.x += (Math.random() - 0.5) * impactShake;
            avatar.position.z += (Math.random() - 0.5) * impactShake;
            avatar.position.y += (Math.random() - 0.5) * impactShake * 0.4;
        }

        // ── Emissive glow tracks tremor intensity ──
        avatar.traverse(child => {
            if (child.isMesh) {
                child.material.emissiveIntensity = 0.15 + tremorFactor * 1.8;
            }
        });
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
