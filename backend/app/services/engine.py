"""
engine.py — upgraded Digital Twin physics engine.
New features:
  - A* pathfinding over a 30x30 grid with obstacle cost map
  - Spatial Fear Memory: past freeze zones raise path costs
  - Fatigue accumulation (rises with activity, lowers at rest)
  - Circadian cycle: slow sine that shifts stress floor & tremor baseline
"""
import random, math, heapq
from datetime import datetime
from collections import defaultdict

# ── Grid constants ─────────────────────────────────────────────────────────
GRID_RES  = 1.0          # each cell = 1 world unit
GRID_W    = 30           # -15 to +15
GRID_HALF = GRID_W // 2  # 15

def _world_to_cell(wx, wz):
    return (int(wx + GRID_HALF), int(wz + GRID_HALF))

def _cell_to_world(cx, cz):
    return (cx - GRID_HALF + 0.5, cz - GRID_HALF + 0.5)

def _clamp_cell(c):
    return max(0, min(GRID_W - 1, c))


class NPC:
    def __init__(self, id, x, z, floor):
        self.id    = id
        self.pos   = {"x": x, "z": z, "floor": floor}
        self.vel   = {"x": 0.0, "z": 0.0}
        self.speed = 0.035
        self.target = {"x": x + random.uniform(-5, 5),
                       "z": z + random.uniform(-5, 5)}
        self.timer = random.randint(100, 300)

    def update(self, engine):
        dx = self.target["x"] - self.pos["x"]
        dz = self.target["z"] - self.pos["z"]
        dist = math.sqrt(dx**2 + dz**2)
        if dist < 0.5 or self.timer <= 0:
            self.target = {
                "x": max(-14, min(14, self.pos["x"] + random.uniform(-8, 8))),
                "z": max(-14, min(14, self.pos["z"] + random.uniform(-8, 8))),
            }
            self.timer = random.randint(150, 400)
            if random.random() < 0.02:
                self.pos["floor"] = 1 if self.pos["floor"] == 0 else 0
        self.timer -= 1
        if dist > 0:
            self.vel["x"] += ((dx / dist) * self.speed - self.vel["x"]) * 0.1
            self.vel["z"] += ((dz / dist) * self.speed - self.vel["z"]) * 0.1
        self.pos["x"] = max(-14.0, min(14.0, self.pos["x"] + self.vel["x"]))
        self.pos["z"] = max(-14.0, min(14.0, self.pos["z"] + self.vel["z"]))


class DigitalTwinEngine:
    # ── Named landmark goals the patient wants to visit ───────────────────
    LANDMARKS = [
        {"name": "Coffee Shop",  "x": -8.0, "z": -8.0, "floor": 0},
        {"name": "Electronics",  "x":  8.0, "z": -8.0, "floor": 0},
        {"name": "Clothing",     "x": -8.0, "z":  8.0, "floor": 0},
        {"name": "Fountain",     "x":  0.0, "z":  0.0, "floor": 0},
        {"name": "Food Court",   "x":  0.0, "z":-10.0, "floor": 1},
        {"name": "Restrooms",    "x":-10.0, "z":  0.0, "floor": 1},
    ]

    def __init__(self):
        self.base_hr       = 70.0
        self.current_hr    = 70.0
        self.base_temp     = 36.6
        self.current_temp  = 36.6
        self.step          = 0
        self.scenario      = "RESTING"
        self.scenario_hr_target     = 65.0
        self.scenario_stress_floor  = 0.0

        # Avatar
        self.pos = {"x": 0.0, "z": 0.0, "floor": 0}
        self.vel = {"x": 0.0, "z": 0.0}
        self.max_speed = 0.04
        self.max_force = 0.004
        self.friction  = 0.88

        # PD state
        self.stress_level       = 0.0
        self.tremor_intensity   = 0.4
        self.is_freezing        = False
        self.freeze_duration    = 0
        self.festination_active = False
        self.festination_timer  = 0
        self.festination_mult   = 1.0
        self.start_hesitation   = 40
        self.gait_phase         = 0.0

        # Manual override
        self.input_vector   = {"x": 0.0, "z": 0.0}
        self.smoothed_input = {"x": 0.0, "z": 0.0}
        self.is_manual      = False

        self.cage_half          = 15.0
        self.avatar_radius      = 0.3
        self.collision_stress_boost = 0.0

        # Obstacles
        self.obstacles = [
            {"floor": 0, "type": "rect",   "x": -8.0, "z": -8.0, "w": 6.0, "d": 6.0, "name": "Coffee Shop"},
            {"floor": 0, "type": "rect",   "x":  8.0, "z": -8.0, "w": 6.0, "d": 4.0, "name": "Electronics"},
            {"floor": 0, "type": "rect",   "x": -8.0, "z":  8.0, "w": 4.0, "d": 6.0, "name": "Clothing"},
            {"floor": 0, "type": "circle", "x":  0.0, "z":  0.0, "radius": 2.0,       "name": "Fountain"},
            {"floor": 1, "type": "rect",   "x":  0.0, "z":-10.0, "w": 12.0, "d": 4.0, "name": "Food Court"},
            {"floor": 1, "type": "rect",   "x":-10.0, "z":  0.0, "w":  3.0, "d": 8.0, "name": "Restrooms"},
            {"floor": 1, "type": "circle", "x":  8.0, "z":  8.0, "radius": 1.5,        "name": "Lounge Pillar"},
        ]
        self.escalators = [{"x": 12.0, "z": 0.0, "radius": 2.0}]

        # Build obstacle cost map (static, computed once)
        self._obstacle_grid = self._build_obstacle_grid()

        # ── NEW: Spatial Fear Memory ────────────────────────────────────────
        # Maps (cx, cz) → cumulative freeze count at that cell
        self.freeze_heatmap: dict[tuple, float] = defaultdict(float)

        # ── NEW: A* navigation ─────────────────────────────────────────────
        self.goal_queue   = list(self.LANDMARKS)          # rotate through
        random.shuffle(self.goal_queue)
        self.current_goal = self.goal_queue[0]
        self.waypoints: list[tuple] = []                  # world (x, z) waypoints
        self.waypoint_timer = 0                           # replan every N steps

        # ── NEW: Fatigue ────────────────────────────────────────────────────
        self.fatigue = 0.0   # 0 = fresh, 1 = exhausted

        # ── NEW: Circadian cycle ────────────────────────────────────────────
        # One full cycle ≈ 12 000 steps (~10 min at 20 Hz)
        self.circadian_period = 12000

        # NPCs
        self.npcs = [
            NPC(id=i,
                x=random.uniform(-10, 10),
                z=random.uniform(-10, 10),
                floor=random.choice([0, 1]))
            for i in range(12)
        ]

    # ── Obstacle grid ──────────────────────────────────────────────────────

    def _build_obstacle_grid(self):
        """Pre-compute a (GRID_W x GRID_W) boolean grid — True = blocked."""
        grid = [[False] * GRID_W for _ in range(GRID_W)]
        for obs in self.obstacles:
            for cx in range(GRID_W):
                for cz in range(GRID_W):
                    wx, wz = _cell_to_world(cx, cz)
                    if obs["type"] == "circle":
                        d = math.sqrt((wx - obs["x"])**2 + (wz - obs["z"])**2)
                        if d < obs["radius"] + self.avatar_radius:
                            grid[cx][cz] = True
                    elif obs["type"] == "rect":
                        if (abs(wx - obs["x"]) < obs["w"] / 2 + self.avatar_radius and
                                abs(wz - obs["z"]) < obs["d"] / 2 + self.avatar_radius):
                            grid[cx][cz] = True
        return grid

    # ── A* pathfinding ─────────────────────────────────────────────────────

    def _astar(self, start_w, goal_w):
        """Return a list of world (x, z) waypoints from start to goal."""
        sc = _world_to_cell(*start_w)
        gc = _world_to_cell(*goal_w)
        sc = (_clamp_cell(sc[0]), _clamp_cell(sc[1]))
        gc = (_clamp_cell(gc[0]), _clamp_cell(gc[1]))

        if self._obstacle_grid[gc[0]][gc[1]]:
            # Goal is blocked — nudge it to nearest free neighbour
            for ddx in range(-3, 4):
                for ddz in range(-3, 4):
                    nc = (_clamp_cell(gc[0] + ddx), _clamp_cell(gc[1] + ddz))
                    if not self._obstacle_grid[nc[0]][nc[1]]:
                        gc = nc
                        break

        def h(c):
            return abs(c[0] - gc[0]) + abs(c[1] - gc[1])

        def fear_cost(c):
            return self.freeze_heatmap.get(c, 0.0) * 2.0  # penalise known freeze zones

        open_heap = [(0, sc)]
        came_from = {}
        g_score   = {sc: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == gc:
                # Reconstruct path
                path = []
                while cur in came_from:
                    path.append(_cell_to_world(*cur))
                    cur = came_from[cur]
                path.reverse()
                return path

            for dx, dz in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nb = (_clamp_cell(cur[0]+dx), _clamp_cell(cur[1]+dz))
                if self._obstacle_grid[nb[0]][nb[1]]:
                    continue
                step_cost = 1.4 if dx != 0 and dz != 0 else 1.0
                tentative  = g_score[cur] + step_cost + fear_cost(nb)
                if tentative < g_score.get(nb, float("inf")):
                    g_score[nb]   = tentative
                    came_from[nb] = cur
                    f = tentative + h(nb)
                    heapq.heappush(open_heap, (f, nb))

        return []  # no path found

    def _replan(self):
        """Recompute waypoints toward current goal."""
        goal = self.current_goal
        if goal["floor"] != self.pos["floor"]:
            # Head to escalator first
            esc = self.escalators[0]
            self.waypoints = self._astar(
                (self.pos["x"], self.pos["z"]),
                (esc["x"], esc["z"]),
            )
        else:
            self.waypoints = self._astar(
                (self.pos["x"], self.pos["z"]),
                (goal["x"], goal["z"]),
            )

    def _advance_goal(self):
        """Move to next landmark in queue."""
        self.goal_queue.append(self.goal_queue.pop(0))
        self.current_goal = self.goal_queue[0]
        self.waypoints = []

    # ── Scenario ───────────────────────────────────────────────────────────

    def set_scenario(self, scenario: str):
        self.scenario = scenario.upper()
        if self.scenario == "RUNNING":
            self.max_speed = 0.09; self.scenario_hr_target = 130.0; self.scenario_stress_floor = 0.25
        elif self.scenario == "SLEEPING":
            self.max_speed = 0.008; self.scenario_hr_target = 52.0; self.scenario_stress_floor = 0.0
        elif self.scenario == "STRESSED":
            self.max_speed = 0.05; self.scenario_hr_target = 98.0; self.scenario_stress_floor = 0.65
        else:
            self.max_speed = 0.04; self.scenario_hr_target = 65.0; self.scenario_stress_floor = 0.0

    def update_controls(self, x: float, z: float):
        self.input_vector = {"x": x, "z": z}
        self.is_manual    = abs(x) > 0.1 or abs(z) > 0.1

    # ── Circadian helper ───────────────────────────────────────────────────

    def _circadian_factor(self):
        """Returns value in [-1, 1]. Negative = night-like (higher stress)."""
        phase = (self.step % self.circadian_period) / self.circadian_period
        return math.sin(2 * math.pi * phase)

    # ── Physics ────────────────────────────────────────────────────────────

    def _update_physics(self):
        # Escalator floor change
        for esc in self.escalators:
            dx = self.pos["x"] - esc["x"]
            dz = self.pos["z"] - esc["z"]
            if math.sqrt(dx**2 + dz**2) < esc["radius"] and not self.is_freezing:
                if random.random() < 0.05:
                    self.pos["floor"] = 1 if self.pos["floor"] == 0 else 0
                    self.vel["x"] *= -1; self.vel["z"] *= -1
                    self.pos["x"] += self.vel["x"] * 10
                    self.pos["z"] += self.vel["z"] * 10
                    self._advance_goal()

        for npc in self.npcs:
            npc.update(self)

        if self.is_freezing:
            self.vel["x"] *= 0.08; self.vel["z"] *= 0.08
            self.freeze_duration -= 1
            if self.freeze_duration <= 0:
                self.is_freezing = False
                self.start_hesitation = 30
            self.pos["x"] += self.vel["x"]; self.pos["z"] += self.vel["z"]
            self._handle_collisions()
            return

        if self.start_hesitation > 0:
            self.start_hesitation -= 1
            return

        lerp = 0.15
        self.smoothed_input["x"] += (self.input_vector["x"] - self.smoothed_input["x"]) * lerp
        self.smoothed_input["z"] += (self.input_vector["z"] - self.smoothed_input["z"]) * lerp

        if self.is_manual:
            self.vel["x"] += self.smoothed_input["x"] * self.max_force
            self.vel["z"] += self.smoothed_input["z"] * self.max_force
        else:
            # ── A* autonomous navigation ───────────────────────────────────
            self.waypoint_timer -= 1
            if self.waypoint_timer <= 0 or not self.waypoints:
                self._replan()
                self.waypoint_timer = 60

            if self.waypoints:
                wp = self.waypoints[0]
                dx = wp[0] - self.pos["x"]
                dz = wp[1] - self.pos["z"]
                dist = math.sqrt(dx**2 + dz**2)
                if dist < 0.8:
                    self.waypoints.pop(0)
                    if not self.waypoints:
                        self._advance_goal()
                elif dist > 0:
                    # Fatigue reduces effective force
                    force = self.max_force * (1.0 - self.fatigue * 0.6)
                    self.vel["x"] += (dx / dist) * force
                    self.vel["z"] += (dz / dist) * force
            else:
                self.vel["x"] += (random.random() - 0.5) * 0.001
                self.vel["z"] += (random.random() - 0.5) * 0.001

        # PD Motor — festination / freeze
        speed = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2)
        if not self.festination_active:
            freeze_prob = 0.0008 + self.fatigue * 0.001
            if speed > self.max_speed * 0.3 and random.random() < freeze_prob:
                self.festination_active = True
                self.festination_timer  = random.randint(50, 120)
                self.festination_mult   = random.uniform(2.2, 3.5)
        else:
            self.festination_timer -= 1
            if self.festination_timer <= 0:
                self.festination_active = False
                self.festination_mult   = 1.0
                self.is_freezing        = True
                self.freeze_duration    = random.randint(25, 60)
                # Record freeze location in heatmap
                cell = _world_to_cell(self.pos["x"], self.pos["z"])
                self.freeze_heatmap[cell] = min(5.0, self.freeze_heatmap[cell] + 1.0)

        self.gait_phase += random.gauss(0.6, 0.15)
        self.vel["x"]   += math.sin(self.gait_phase) * 0.0004
        self.vel["z"]   += math.sin(self.gait_phase) * 0.0004 * random.uniform(-1, 1)

        effective_max = self.max_speed * self.festination_mult
        speed = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2)
        if speed > effective_max:
            self.vel["x"] = (self.vel["x"] / speed) * effective_max
            self.vel["z"] = (self.vel["z"] / speed) * effective_max

        self.vel["x"] *= self.friction; self.vel["z"] *= self.friction
        self.pos["x"] += self.vel["x"]; self.pos["z"] += self.vel["z"]
        self._handle_collisions()

    # ── Hazard proximity ───────────────────────────────────────────────────

    def _get_hazard_proximity(self):
        max_p = 0.0
        for h in self.obstacles:
            if h["floor"] != self.pos["floor"]: continue
            if h["type"] == "circle":
                d = math.sqrt((self.pos["x"]-h["x"])**2 + (self.pos["z"]-h["z"])**2)
                awareness = h["radius"] + 3.0
                if d < awareness: max_p = max(max_p, (awareness - d) / awareness)
            elif h["type"] == "rect":
                dx = max(abs(self.pos["x"] - h["x"]) - h["w"]/2, 0)
                dz = max(abs(self.pos["z"] - h["z"]) - h["d"]/2, 0)
                d  = math.sqrt(dx**2 + dz**2)
                if d < 3.0: max_p = max(max_p, (3.0 - d) / 3.0)
        for npc in self.npcs:
            if npc.pos["floor"] != self.pos["floor"]: continue
            d = math.sqrt((self.pos["x"]-npc.pos["x"])**2 + (self.pos["z"]-npc.pos["z"])**2)
            if d < 4.0: max_p = max(max_p, (4.0 - d) / 4.0)
        return min(max_p, 1.0)

    # ── Physiology ─────────────────────────────────────────────────────────

    def _update_physiology(self):
        hazard_prox  = self._get_hazard_proximity()
        raw_speed    = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2)
        movement_load = min(raw_speed / max(self.max_speed, 0.001), 1.0)
        circ = self._circadian_factor()  # -1 (night) to +1 (day)

        # Fatigue: rises with movement, falls at rest
        if movement_load > 0.3:
            self.fatigue = min(1.0, self.fatigue + 0.0003 * movement_load)
        else:
            self.fatigue = max(0.0, self.fatigue - 0.0001)

        self.collision_stress_boost *= 0.97
        # Night-phase (circ < 0) raises baseline stress
        circadian_stress = max(0.0, -circ * 0.15)
        target_stress = max(
            self.scenario_stress_floor,
            hazard_prox * 0.8 + movement_load * 0.2
            + self.collision_stress_boost
            + self.fatigue * 0.1
            + circadian_stress,
        )
        self.stress_level += (target_stress - self.stress_level) * 0.1

        hr_goal = self.scenario_hr_target + (self.stress_level * 35.0) + (movement_load * 30.0)
        self.current_hr += (hr_goal - self.current_hr) * 0.10

        temp_goal = self.base_temp + (movement_load * 0.9) + (self.stress_level * 0.6)
        self.current_temp += (temp_goal - self.current_temp) * 0.07

        # Tremor = resting component + stress + fatigue + circadian (worse at night)
        circadian_tremor = max(0.0, -circ * 0.1)
        resting_comp = 0.45 * max(0.0, 1.0 - movement_load * 0.7)
        self.tremor_intensity = (resting_comp + self.stress_level * 0.25
                                 + self.collision_stress_boost
                                 + self.fatigue * 0.1
                                 + circadian_tremor)

        # Freeze probability boosted by fatigue and night cycle
        freeze_prob = 0.002 + hazard_prox * 0.06 + self.fatigue * 0.01 + max(0, -circ) * 0.005
        if not self.is_freezing and random.random() < freeze_prob:
            self.is_freezing    = True
            self.freeze_duration = random.randint(30, 90)
            cell = _world_to_cell(self.pos["x"], self.pos["z"])
            self.freeze_heatmap[cell] = min(5.0, self.freeze_heatmap[cell] + 0.5)

    # ── Collisions ─────────────────────────────────────────────────────────

    def _handle_collisions(self):
        ch = self.cage_half; ar = self.avatar_radius
        if self.pos["x"] < -ch+ar: self.pos["x"] = -ch+ar; self.vel["x"] *= -0.2
        elif self.pos["x"] > ch-ar: self.pos["x"] = ch-ar; self.vel["x"] *= -0.2
        if self.pos["z"] < -ch+ar: self.pos["z"] = -ch+ar; self.vel["z"] *= -0.2
        elif self.pos["z"] > ch-ar: self.pos["z"] = ch-ar; self.vel["z"] *= -0.2

        for obs in self.obstacles:
            if obs["floor"] != self.pos["floor"]: continue
            if obs["type"] == "circle":
                dx = self.pos["x"] - obs["x"]; dz = self.pos["z"] - obs["z"]
                dist = math.sqrt(dx**2 + dz**2)
                if dist < obs["radius"] + ar:
                    self.pos["x"] += (dx/dist) * (obs["radius"]+ar - dist)
                    self.pos["z"] += (dz/dist) * (obs["radius"]+ar - dist)
                    self.collision_stress_boost = min(0.9, self.collision_stress_boost + 0.3)
            elif obs["type"] == "rect":
                hw = obs["w"]/2; hd = obs["d"]/2
                dx = self.pos["x"] - obs["x"]; dz = self.pos["z"] - obs["z"]
                if abs(dx) < hw+ar and abs(dz) < hd+ar:
                    ox = hw+ar - abs(dx); oz = hd+ar - abs(dz)
                    if ox < oz:
                        self.pos["x"] += ox if dx > 0 else -ox; self.vel["x"] *= -0.5
                    else:
                        self.pos["z"] += oz if dz > 0 else -oz; self.vel["z"] *= -0.5
                    self.collision_stress_boost = min(0.9, self.collision_stress_boost + 0.3)

    # ── Generate reading ───────────────────────────────────────────────────

    def generate_reading(self):
        self.step += 1
        self._update_physics()
        self._update_physiology()

        act = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2) * 2.5
        circ = self._circadian_factor()

        return {
            "heart_rate":        round(self.current_hr + (random.random()-0.5)*2.0, 1),
            "temperature":       round(self.current_temp + (random.random()-0.5)*0.1, 2),
            "activity_level":    round(act, 2),
            "position":          {"x": round(self.pos["x"], 3),
                                   "z": round(self.pos["z"], 3),
                                   "floor": self.pos["floor"]},
            "hazard_proximity":  round(self._get_hazard_proximity(), 2),
            "tremor_intensity":  round(self.tremor_intensity, 3),
            "is_freezing":       self.is_freezing,
            "festination_active": self.festination_active,
            "stress_level":      round(self.stress_level, 2),
            "fatigue":           round(self.fatigue, 2),
            "circadian":         round(circ, 2),
            "current_goal":      self.current_goal["name"],
            "breath_phase":      round((math.sin(self.step * 0.1) + 1.0) * 0.5, 2),
            "sway":              round(math.sin(self.step * 0.05) * self.tremor_intensity, 2),
            "timestamp":         datetime.utcnow().isoformat(),
            "npcs":              [{"id": n.id,
                                   "x": round(n.pos["x"], 2),
                                   "z": round(n.pos["z"], 2),
                                   "floor": n.pos["floor"]} for n in self.npcs],
            "freeze_heatmap":    [{"x": _cell_to_world(cx, cz)[0],
                                   "z": _cell_to_world(cx, cz)[1],
                                   "v": v}
                                  for (cx, cz), v in self.freeze_heatmap.items()],
        }


engine = DigitalTwinEngine()
