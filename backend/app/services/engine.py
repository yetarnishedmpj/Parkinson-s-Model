import random, math, heapq
from datetime import datetime
from collections import defaultdict
import numpy as np
from functools import lru_cache

# ── Grid constants ─────────────────────────────────────
GRID_RES  = 1.0          # each cell = 1 world unit
GRID_W    = 30           # -15 to +15
GRID_HALF = GRID_W // 2  # 15

def _world_to_cell(wx, wz):
    return (int(wx + GRID_HALF), int(wz + GRID_HALF))

def _cell_to_world(cx, cz):
    return (cx - GRID_HALF + 0.5, cz - GRID_HALF + 0.5)

def _clamp_cell(c):
    return max(0, min(GRID_W - 1, c))


# Spatial partitioning constants for obstacle lookup
CELL_SIZE = 1.0  # size of each spatial cell in world units

class NPC:
    def __init__(self, id, x, z, floor):
        self.id = id
        # position and velocity as numpy arrays [x, z]
        self.pos = np.array([x, z], dtype=np.float64)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)
        self.speed = 0.035
        self.target = np.array([
            x + random.uniform(-5, 5),
            z + random.uniform(-5, 5)
        ], dtype=np.float64)
        self.floor = floor
        self.timer = random.randint(100, 300)

    def update(self, engine):
        # engine is DigitalTwinEngine instance (unused but kept for signature)
        dx = self.target[0] - self.pos[0]
        dz = self.target[1] - self.pos[1]
        dist = math.sqrt(dx*dx + dz*dz)
        if dist < 0.5 or self.timer <= 0:
            self.target = np.array([
                max(-14, min(14, self.pos[0] + random.uniform(-8, 8))),
                max(-14, min(14, self.pos[1] + random.uniform(-8, 8)))
            ], dtype=np.float64)
            self.timer = random.randint(150, 400)
            if random.random() < 0.02:
                self.floor = 1 if self.floor == 0 else 0
        self.timer -= 1
        if dist > 0:
            # velocity update: vel += ((dx/dist)*speed - vel) * 0.1
            self.vel += ((np.array([dx, dz], dtype=np.float64) / dist) * self.speed - self.vel) * 0.1
        # position update
        self.pos += self.vel
        # clamp position to bounds
        self.pos[0] = max(-14.0, min(14.0, self.pos[0]))
        self.pos[1] = max(-14.0, min(14.0, self.pos[1]))

class DigitalTwinEngine:
    LANDMARKS = [
        {"name": "Coffee Shop",  "x": -8.0, "z": -8.0, "floor": 0},
        {"name": "Electronics",  "x":  8.0, "z": -8.0, "floor": 0},
        {"name": "Clothing",     "x": -8.0, "z":  8.0, "floor": 0},
        {"name": "Fountain",     "x":  0.0, "z":  0.0, "floor": 0},
        {"name": "Food Court",   "x":  0.0, "z":-10.0, "floor": 1},
        {"name": "Restrooms",    "x":-10.0, "z":  0.0, "floor": 1},
    ]

    def __init__(self):
        # Biometrics
        self.base_hr       = 70.0
        self.current_hr    = 70.0
        self.base_temp     = 36.6
        self.current_temp  = 36.6
        self.step          = 0
        self.scenario      = "RESTING"

        # Scenario-driven physiological baselines
        self.scenario_hr_target     = 65.0
        self.scenario_stress_floor  = 0.0

        # Physics & Avatar
        self.pos = np.array([0.0, 0.0], dtype=np.float64)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)
        self.floor = 0
        self.max_speed = 0.04
        self.max_force = 0.004
        self.friction  = 0.88

        # PD state
        self.stress_level       = 0.0
        self.fatigue            = 0.0
        self.tremor_intensity   = 0.4
        self.is_freezing        = False
        self.freeze_duration    = 0
        self.festination_active = False
        self.festination_timer  = 0
        self.festination_mult   = 1.0
        self.start_hesitation   = 40
        self.gait_phase         = 0.0

        # Control Input
        self.input_vector   = {"x": 0.0, "z": 0.0}
        self.smoothed_input = {"x": 0.0, "z": 0.0}
        self.is_manual      = False

        self.cage_half          = 15.0
        self.avatar_radius      = 0.3
        self.collision_stress_boost = 0.0

        # Detailed Hazards (Combined with floors)
        self.obstacles = [
            {"floor": 0, "type": "doorway",      "x":  0.0, "z": -8.0, "w": 6.0, "d": 6.0, "radius": 1.8, "name": "Coffee Shop"},
            {"floor": 0, "type": "rect",         "x":  8.0, "z": -8.0, "w": 6.0, "d": 4.0, "radius": 2.0, "name": "Electronics"},
            {"floor": 0, "type": "rect",         "x": -8.0, "z":  8.0, "w": 4.0, "d": 6.0, "radius": 1.8, "name": "Clothing"},
            {"floor": 0, "type": "circle",       "x":  0.0, "z":  0.0, "radius": 2.0, "name": "Fountain"},
            {"floor": 1, "type": "narrow_hall",  "x":  0.0, "z":-10.0, "w": 12.0, "d": 4.0, "radius": 1.8, "name": "Food Court"},
            {"floor": 1, "type": "doorway",      "x":-10.0, "z":  0.0, "w":  3.0, "d": 8.0, "radius": 1.5, "name": "Restrooms"},
            {"floor": 1, "type": "circle",       "x":  8.0, "z":  8.0, "radius": 1.5, "name": "Lounge Pillar"},
        ]
        self.escalators = [{"x": 12.0, "z": 0.0, "radius": 2.0}]

        self._obstacle_grid = self._build_obstacle_grid()

        # Spatial Fear Memory
        self.freeze_heatmap: dict[tuple, float] = defaultdict(float)

        # A* navigation
        self.goal_queue   = list(self.LANDMARKS)
        random.shuffle(self.goal_queue)
        self.current_goal = self.goal_queue[0]
        self.waypoints: list[tuple] = []
        self.waypoint_timer = 0

        # Circadian cycle
        self.circadian_period = 12000

        # NPCs
        self.npcs = [
            NPC(id=i, x=random.uniform(-10, 10), z=random.uniform(-10, 10), floor=random.choice([0, 1]))
            for i in range(12)
        ]

    def _build_obstacle_grid(self):
        grid = [[False] * GRID_W for _ in range(GRID_W)]
        for obs in self.obstacles:
            for cx in range(GRID_W):
                for cz in range(GRID_W):
                    wx, wz = _cell_to_world(cx, cz)
                    if obs["type"] == "circle":
                        d = math.sqrt((wx - obs["x"])**2 + (wz - obs["z"])**2)
                        if d < obs["radius"] + self.avatar_radius:
                            grid[cx][cz] = True
                    elif obs["type"] in ["rect", "doorway", "narrow_hall"]:
                        w = obs.get("w", obs["radius"] * 2)
                        d_val = obs.get("d", obs["radius"] * 2)
                        if (abs(wx - obs["x"]) < w / 2 + self.avatar_radius and
                                abs(wz - obs["z"]) < d_val / 2 + self.avatar_radius):
                            grid[cx][cz] = True
        return grid

    @lru_cache(maxsize=128)
def _astar(self, start_w, goal_w):
        sc = _world_to_cell(*start_w)
        gc = _world_to_cell(*goal_w)
        sc = (_clamp_cell(sc[0]), _clamp_cell(sc[1]))
        gc = (_clamp_cell(gc[0]), _clamp_cell(gc[1]))

        if self._obstacle_grid[gc[0]][gc[1]]:
            for ddx in range(-3, 4):
                for ddz in range(-3, 4):
                    nc = (_clamp_cell(gc[0] + ddx), _clamp_cell(gc[1] + ddz))
                    if not self._obstacle_grid[nc[0]][nc[1]]:
                        gc = nc
                        break

        def h(c):
            return abs(c[0] - gc[0]) + abs(c[1] - gc[1])

        def fear_cost(c):
            return self.freeze_heatmap.get(c, 0.0) * 2.0

        open_heap = [(0, sc)]
        came_from = {}
        g_score   = {sc: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == gc:
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
        return []

    def _replan(self):
        goal = self.current_goal
        if goal["floor"] != self.floor:
            esc = self.escalators[0]
            self.waypoints = self._astar((self.pos[0], self.pos[1]), (esc["x"], esc["z"]))
        else:
            self.waypoints = self._astar((self.pos[0], self.pos[1]), (goal["x"], goal["z"]))

    def _advance_goal(self):
        self.goal_queue.append(self.goal_queue.pop(0))
        self.current_goal = self.goal_queue[0]
        self.waypoints = []

    def set_scenario(self, scenario: str):
        self.scenario = scenario.upper()
        if self.scenario == "RUNNING":
            self.max_speed = 0.09; self.scenario_hr_target = 130.0; self.scenario_stress_floor = 0.25
        elif self.scenario == "SLEEPING":
            self.max_speed = 0.008; self.scenario_hr_target = 52.0; self.scenario_stress_floor = 0.0
            self.stress_level = 0.0; self.is_freezing = False
        elif self.scenario == "STRESSED":
            self.max_speed = 0.05; self.scenario_hr_target = 98.0; self.scenario_stress_floor = 0.65
            self.stress_level = 0.65
        else:
            self.max_speed = 0.04; self.scenario_hr_target = 65.0; self.scenario_stress_floor = 0.0

    def update_controls(self, x: float, z: float):
        self.input_vector = {"x": x, "z": z}
        self.is_manual    = abs(x) > 0.1 or abs(z) > 0.1

    def _circadian_factor(self):
        phase = (self.step % self.circadian_period) / self.circadian_period
        return math.sin(2 * math.pi * phase)

    def _apply_force(self, fx, fz):
        self.vel[0] += fx
        self.vel[1] += fz

    def _update_physics(self):
        for esc in self.escalators:
            dx = self.pos[0] - esc["x"]
            dz = self.pos[1] - esc["z"]
            if math.sqrt(dx**2 + dz**2) < esc["radius"] and not self.is_freezing:
                if random.random() < 0.05:
                    self.floor = 1 if self.floor == 0 else 0
                    self.vel[0] *= -1; self.vel[1] *= -1
                    self.pos[0] += self.vel[0] * 10
                    self.pos[1] += self.vel[1] * 10
                    self._advance_goal()

        for npc in self.npcs:
            npc.update(self)

        if self.is_freezing:
            self.vel[0] *= 0.08; self.vel[1] *= 0.08
            self.freeze_duration -= 1
            if self.freeze_duration <= 0:
                self.is_freezing = False
                self.start_hesitation = random.randint(15, 40)
            self.pos[0] += self.vel[0]; self.pos[1] += self.vel[1]
            self._handle_collisions()
            return

        if self.start_hesitation > 0:
            self.start_hesitation -= 1
            self.vel[0] *= 0.5; self.vel[1] *= 0.5
            self.pos[0] += self.vel[0]; self.pos[1] += self.vel[1]
            self._handle_collisions()
            return

        lerp_factor = 0.15
        self.smoothed_input["x"] += (self.input_vector["x"] - self.smoothed_input["x"]) * lerp_factor
        self.smoothed_input["z"] += (self.input_vector["z"] - self.smoothed_input["z"]) * lerp_factor

        if self.is_manual:
            self.vel[0] += self.smoothed_input["x"] * self.max_force
            self.vel[1] += self.smoothed_input["z"] * self.max_force
        else:
            self.waypoint_timer -= 1
            if self.waypoint_timer <= 0 or not self.waypoints:
                self._replan()
                self.waypoint_timer = 60

            if self.waypoints:
                wp = self.waypoints[0]
                dx = wp[0] - self.pos[0]; dz = wp[1] - self.pos[1]
                dist = math.sqrt(dx**2 + dz**2)
                if dist < 0.8:
                    self.waypoints.pop(0)
                    if not self.waypoints:
                        self._advance_goal()
                elif dist > 0:
                    force = self.max_force * (1.0 - self.fatigue * 0.6)
                    rx, rz = self._get_obstacle_repulsion()
                    self._apply_force((dx/dist)*force + rx, (dz/dist)*force + rz)

        self._apply_parkinson_motor()

        effective_max = self.max_speed * self.festination_mult
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        if speed > effective_max:
            self.vel[0] = (self.vel[0] / speed) * effective_max
            self.vel[1] = (self.vel[1] / speed) * effective_max

        self.vel[0] *= self.friction; self.vel[1] *= self.friction
        self.pos[0] += self.vel[0]; self.pos[1] += self.vel[1]
        self._handle_collisions()

    def _apply_parkinson_motor(self):
        speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)

        if not self.festination_active:
            if not self.is_manual and speed > self.max_speed * 0.3 and random.random() < 0.0008:
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
                    self.collision_stress_boost = min(0.9, self.collision_stress_boost + 0.5)
                    cell = _world_to_cell(self.pos[0], self.pos[1])
                    self.freeze_heatmap[cell] = min(5.0, self.freeze_heatmap[cell] + 1.0)

        self.gait_phase += random.gauss(0.6, 0.15)
        shuffle_noise = math.sin(self.gait_phase) * 0.0004
        self.vel[0] += shuffle_noise
        self.vel[1] += shuffle_noise * random.uniform(-1, 1)

        if speed > 0.005 and random.random() < 0.04:
            perp_x = -self.vel[1]
            perp_z =  self.vel[0]
            pmag = math.sqrt(perp_x**2 + perp_z**2)
            if pmag > 0:
                drift = random.gauss(0, 0.0006)
                self.vel[0] += (perp_x / pmag) * drift
                self.vel[1] += (perp_z / pmag) * drift

    def _get_obstacle_repulsion(self):
        rx, rz = 0.0, 0.0
        for obs in self.obstacles:
            if obs["floor"] != self.floor: continue
            dx = self.pos[0] - obs["x"]
            dz = self.pos[1] - obs["z"]
            dist = math.sqrt(dx * dx + dz * dz)
            influence = obs.get("radius", 2.0) + 2.5
            if 0.01 < dist < influence:
                strength = ((influence - dist) / influence) ** 2
                rx += (dx / dist) * strength * self.max_force * 3.0
                rz += (dz / dist) * strength * self.max_force * 3.0
        return rx, rz

    def _get_hazard_proximity(self):
        max_p = 0.0
        for h in self.obstacles:
            if h["floor"] != self.floor: continue
            if h["type"] == "circle":
                d = math.sqrt((self.pos[0]-h["x"])**2 + (self.pos[1]-h["z"])**2)
                awareness = h["radius"] + 3.0
                if d < awareness: max_p = max(max_p, (awareness - d) / awareness)
            elif h["type"] in ["rect", "doorway", "narrow_hall"]:
                w = h.get("w", h["radius"] * 2)
                d_val = h.get("d", h["radius"] * 2)
                dx = max(abs(self.pos[0] - h["x"]) - w/2, 0)
                dz = max(abs(self.pos[1] - h["z"]) - d_val/2, 0)
                d  = math.sqrt(dx**2 + dz**2)
                if d < 3.0: max_p = max(max_p, (3.0 - d) / 3.0)
        for npc in self.npcs:
            if npc.floor != self.floor: continue
            dx = self.pos[0] - npc.pos[0]
            dz = self.pos[1] - npc.pos[1]
            d = math.sqrt(dx*dx + dz*dz)
            if d < 4.0: max_p = max(max_p, (4.0 - d) / 4.0)
        return min(max_p, 1.0)

    def _update_physiology(self):
        hazard_prox   = self._get_hazard_proximity()
        raw_speed     = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
        movement_load = min(raw_speed / max(self.max_speed, 0.001), 1.0)
        circ          = self._circadian_factor()

        if movement_load > 0.3:
            self.fatigue = min(1.0, self.fatigue + 0.0003 * movement_load)
        else:
            self.fatigue = max(0.0, self.fatigue - 0.0001)

        self.collision_stress_boost *= 0.97
        circadian_stress = max(0.0, -circ * 0.15)
        target_stress = max(
            self.scenario_stress_floor,
            hazard_prox * 0.8 + movement_load * 0.2 + self.collision_stress_boost + self.fatigue * 0.1 + circadian_stress
        )
        self.stress_level += (target_stress - self.stress_level) * 0.1

        hr_goal = self.scenario_hr_target + (self.stress_level * 35.0) + (movement_load * 30.0)
        self.current_hr += (hr_goal - self.current_hr) * 0.10

        temp_goal = self.base_temp + (movement_load * 0.9) + (self.stress_level * 0.6)
        self.current_temp += (temp_goal - self.current_temp) * 0.07

        circadian_tremor = max(0.0, -circ * 0.1)
        resting_comp = 0.45 * max(0.0, 1.0 - movement_load * 0.7)
        self.tremor_intensity = (resting_comp + self.stress_level * 0.25 + self.collision_stress_boost + self.fatigue * 0.1 + circadian_tremor)

        if self.scenario != "SLEEPING" and not self.is_freezing:
            doorway_prox = 0.0
            for h in self.obstacles:
                if h["floor"] != self.floor: continue
                if h["type"] in ("doorway", "narrow_hall"):
                    d = math.sqrt((self.pos[0]-h["x"])**2 + (self.pos[1]-h["z"])**2)
                    awareness = h["radius"] + 3.5
                    if d < awareness:
                        doorway_prox = max(doorway_prox, (awareness - d) / awareness)

            freeze_prob = 0.002 + hazard_prox * 0.06 + doorway_prox * 0.12 + self.fatigue * 0.01 + max(0, -circ) * 0.005
            if random.random() < freeze_prob:
                self.is_freezing = True
                self.freeze_duration = random.randint(30, 90)
                cell = _world_to_cell(self.pos[0], self.pos[1])
                self.freeze_heatmap[cell] = min(5.0, self.freeze_heatmap[cell] + 0.5)

    def _handle_collisions(self):
        ch = self.cage_half; ar = self.avatar_radius
        if self.pos[0] < -ch+ar: self.pos[0] = -ch+ar; self.vel[0] = abs(self.vel[0]) * 0.25
        elif self.pos[0] > ch-ar: self.pos[0] = ch-ar; self.vel[0] = -abs(self.vel[0]) * 0.25
        if self.pos[1] < -ch+ar: self.pos[1] = -ch+ar; self.vel[1] = abs(self.vel[1]) * 0.25
        elif self.pos[1] > ch-ar: self.pos[1] = ch-ar; self.vel[1] = -abs(self.vel[1]) * 0.25

        for obs in self.obstacles:
            if obs["floor"] != self.floor: continue

            if obs["type"] == "circle":
                dx = self.pos[0] - obs["x"]; dz = self.pos[1] - obs["z"]
                dist_sq = dx**2 + dz**2
                min_dist = obs["radius"] + ar
                if 0.0001 < dist_sq < min_dist**2:
                    dist = math.sqrt(dist_sq)
                    nx, nz = dx/dist, dz/dist
                    self.pos[0] += nx * (min_dist - dist)
                    self.pos[1] += nz * (min_dist - dist)
                    dot = self.vel[0]*nx + self.vel[1]*nz
                    if dot < 0:
                        self.vel[0] -= 1.25 * dot * nx; self.vel[1] -= 1.25 * dot * nz
                        self.collision_stress_boost = min(0.9, self.collision_stress_boost + 0.35)

            elif obs["type"] in ["rect", "doorway", "narrow_hall"]:
                w = obs.get("w", obs["radius"] * 2)
                d_val = obs.get("d", obs["radius"] * 2)
                hw, hd = w/2, d_val/2
                dx = self.pos[0] - obs["x"]; dz = self.pos[1] - obs["z"]
                if abs(dx) < hw+ar and abs(dz) < hd+ar:
                    ox = hw+ar - abs(dx); oz = hd+ar - abs(dz)
                    if ox < oz:
                        self.pos[0] += ox if dx > 0 else -ox; self.vel[0] *= -0.5
                    else:
                        self.pos[1] += oz if dz > 0 else -oz; self.vel[1] *= -0.5
                    self.collision_stress_boost = min(0.9, self.collision_stress_boost + 0.35)

engine = DigitalTwinEngine()