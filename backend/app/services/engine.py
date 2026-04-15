
import random
import math
from datetime import datetime

class DigitalTwinEngine:
    def __init__(self):
        # Biometrics
        self.base_hr = 70.0
        self.current_hr = 70.0
        self.base_temp = 36.6
        self.current_temp = 36.6
        self.step = 0
        self.scenario = "RESTING"

        # Scenario-driven physiological baselines (injected directly)
        self.scenario_hr_target = 65.0    # bpm floor set by scenario
        self.scenario_stress_floor = 0.0  # minimum stress level for this scenario

        # Physics
        self.pos = {"x": 5.0, "z": 5.0}
        self.vel = {"x": 0.0, "z": 0.0}
        self.max_speed = 0.04   # units/tick @ 20Hz ≈ 0.8 m/s (realistic resting walk)
        self.max_force = 0.004  # gentle acceleration — no sudden sprinting
        self.friction  = 0.88   # slightly more drag so stops feel natural

        # Goals & Intent (spread across cage, clear of obstacle centres)
        self.current_goal = {"x": 0.0, "z": 0.0}
        self.goals = [
            {"x":  11.0, "z":  11.0, "name": "Corner NE"},
            {"x": -11.0, "z":  -1.0, "name": "West Side"},
            {"x":   0.0, "z": -11.0, "name": "South End"},
            {"x":   0.0, "z":   0.0, "name": "Center"},
            {"x":  -9.0, "z":   9.0, "name": "Corner NW"},
            {"x":   9.0, "z":  -9.0, "name": "Corner SE"},
        ]
        self.next_goal_timer = 100

        # Brain, Stress & PD Motor State
        self.stress_level      = 0.0
        self.fatigue           = 0.0
        self.tremor_intensity  = 0.4    # starts high (resting tremor)
        self.is_freezing       = False
        self.freeze_duration   = 0

        # Parkinson's-specific motor disturbances
        self.festination_active = False  # involuntary forward acceleration episode
        self.festination_timer  = 0
        self.festination_mult   = 1.0   # speed multiplier during festination
        self.start_hesitation   = 40    # ticks of initiation delay on startup
        self.gait_phase         = 0.0   # internal shuffling rhythm

        # Control Input
        self.input_vector = {"x": 0.0, "z": 0.0}
        self.smoothed_input = {"x": 0.0, "z": 0.0}
        self.is_manual = False

        # Hazards — 8 rigid obstacles (radius = physical collision boundary)
        self.obstacles = [
            {"type": "doorway",      "x":  0.0, "z": -8.0, "radius": 1.8},
            {"type": "stairs",       "x":  8.0, "z":  8.0, "radius": 2.0},
            {"type": "narrow_hall",  "x": -8.0, "z":  2.0, "radius": 1.8},
            {"type": "pillar",       "x":  4.0, "z": -3.0, "radius": 1.2},
            {"type": "furniture",    "x": -5.0, "z": -6.0, "radius": 1.5},
            {"type": "wall_block",   "x":  7.0, "z":  0.0, "radius": 1.2},
            {"type": "table",        "x": -2.0, "z":  7.0, "radius": 1.3},
            {"type": "cabinet",      "x":  2.0, "z": 11.0, "radius": 1.3},
        ]
        self.cage_half    = 13.0  # Hard boundary — avatar confined to ±13 units
        self.avatar_radius = 0.3  # Collision sphere radius
        self.collision_stress_boost = 0.0  # Acute spike injected on impact, decays over time

    def set_scenario(self, scenario: str):
        self.scenario = scenario.upper()

        # Each scenario sets a meaningful baseline for ALL physiological systems
        if self.scenario == "RUNNING":
            self.max_speed = 0.09   # ~1.8 m/s — brisk walk / light jog
            self.scenario_hr_target = 130.0
            self.scenario_stress_floor = 0.25
        elif self.scenario == "SLEEPING":
            self.max_speed = 0.008  # barely any movement
            self.scenario_hr_target = 52.0
            self.scenario_stress_floor = 0.0
            self.stress_level = 0.0
            self.is_freezing = False
        elif self.scenario == "STRESSED":
            self.max_speed = 0.05   # ~1.0 m/s — anxious shuffle
            self.scenario_hr_target = 98.0
            self.scenario_stress_floor = 0.65
            self.stress_level = 0.65
        else:  # RESTING
            self.max_speed = 0.04   # ~0.8 m/s — slow cautious walk
            self.scenario_hr_target = 65.0
            self.scenario_stress_floor = 0.0

    def update_controls(self, x: float, z: float):
        self.input_vector = {"x": x, "z": z}
        self.is_manual = abs(x) > 0.1 or abs(z) > 0.1

    def _apply_force(self, fx, fz):
        self.vel["x"] += fx
        self.vel["z"] += fz

    def _get_steering(self, target_x, target_z):
        desired_x = target_x - self.pos["x"]
        desired_z = target_z - self.pos["z"]
        dist = math.sqrt(desired_x**2 + desired_z**2)
        if dist < 0.1: return 0, 0
        desired_x = (desired_x / dist) * self.max_speed
        desired_z = (desired_z / dist) * self.max_speed
        steer_x = desired_x - self.vel["x"]
        steer_z = desired_z - self.vel["z"]
        force_mag = math.sqrt(steer_x**2 + steer_z**2)
        if force_mag > self.max_force:
            ratio = self.max_force / force_mag
            steer_x *= ratio
            steer_z *= ratio
        return steer_x, steer_z

    def _update_physics(self):
        if self.is_freezing:
            # During freeze: damp velocity but still resolve collisions (no wall tunnelling)
            self.vel["x"] *= 0.08
            self.vel["z"] *= 0.08
            self.freeze_duration -= 1
            if self.freeze_duration <= 0:
                self.is_freezing = False
                self.start_hesitation = random.randint(15, 40)  # must restart after freeze
            self.pos["x"] += self.vel["x"]
            self.pos["z"] += self.vel["z"]
            self._handle_collisions()
            return

        # Start hesitation: patient struggles to initiate movement
        if self.start_hesitation > 0:
            self.start_hesitation -= 1
            self.vel["x"] *= 0.5
            self.vel["z"] *= 0.5
            self.pos["x"] += self.vel["x"]
            self.pos["z"] += self.vel["z"]
            self._handle_collisions()
            return

        # Smoothed Manual Input
        lerp_factor = 0.15
        self.smoothed_input["x"] += (self.input_vector["x"] - self.smoothed_input["x"]) * lerp_factor
        self.smoothed_input["z"] += (self.input_vector["z"] - self.smoothed_input["z"]) * lerp_factor

        if self.is_manual:
            self.vel["x"] += self.smoothed_input["x"] * self.max_force
            self.vel["z"] += self.smoothed_input["z"] * self.max_force
        else:
            self.next_goal_timer -= 1
            if self.next_goal_timer <= 0:
                goal = random.choice(self.goals)
                self.current_goal = {"x": goal["x"], "z": goal["z"]}
                self.next_goal_timer = random.randint(300, 600)
                self.start_hesitation = random.randint(10, 30)  # hesitation on each new goal

            sx, sz = self._get_steering(self.current_goal["x"], self.current_goal["z"])
            rx, rz = self._get_obstacle_repulsion()
            self._apply_force(sx + rx, sz + rz)

        # Parkinson's motor disturbances (shuffling, festination, drift)
        self._apply_parkinson_motor()

        # Speed limiter — festination_mult raises the ceiling during episodes
        effective_max = self.max_speed * self.festination_mult
        speed = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2)
        if speed > effective_max:
            self.vel["x"] = (self.vel["x"] / speed) * effective_max
            self.vel["z"] = (self.vel["z"] / speed) * effective_max

        # Friction & Position
        self.vel["x"] *= self.friction
        self.vel["z"] *= self.friction
        self.pos["x"] += self.vel["x"]
        self.pos["z"] += self.vel["z"]
        self._handle_collisions()

    def _update_physiology(self):
        hazard_prox = self._get_hazard_proximity()
        raw_speed = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2)
        movement_load = min(raw_speed / max(self.max_speed, 0.001), 1.0)

        # Decay collision spike
        self.collision_stress_boost *= 0.978

        # 1. Stress
        target_stress = max(
            self.scenario_stress_floor,
            hazard_prox * 0.65 + movement_load * 0.25 + self.collision_stress_boost
        )
        self.stress_level += (target_stress - self.stress_level) * 0.12
        self.stress_level = max(self.stress_level, self.scenario_stress_floor)

        # 2. HR
        hr_goal = self.scenario_hr_target + (self.stress_level * 35.0) + (movement_load * 30.0)
        self.current_hr += (hr_goal - self.current_hr) * 0.10

        # 3. Temperature
        temp_goal = self.base_temp + (movement_load * 0.9) + (self.stress_level * 0.6)
        self.current_temp += (temp_goal - self.current_temp) * 0.07

        # 4. Tremor — Parkinson's RESTING tremor model
        #    Tremor is WORST when still and REDUCES during intentional movement
        #    (opposite of essential tremor — clinically accurate for PD)
        resting_component  = 0.45 * max(0.0, 1.0 - movement_load * 0.7)
        exertion_component = self.stress_level * 0.25
        impact_component   = self.collision_stress_boost * 0.6
        festination_boost  = 0.3 if self.festination_active else 0.0
        self.tremor_intensity = resting_component + exertion_component + impact_component + festination_boost
        if self.scenario == "SLEEPING":
            self.tremor_intensity *= 0.05

        # 5. Freezing — doorways trigger it 4× more (hallmark PD symptom)
        if self.scenario != "SLEEPING" and not self.is_freezing:
            # Find nearest doorway-type obstacle for extra freeze probability
            doorway_prox = 0.0
            for h in self.obstacles:
                if h["type"] in ("doorway", "narrow_hall"):
                    d = math.sqrt((self.pos["x"]-h["x"])**2 + (self.pos["z"]-h["z"])**2)
                    awareness = h["radius"] + 3.5
                    if d < awareness:
                        doorway_prox = max(doorway_prox, (awareness - d) / awareness)
            freeze_chance = 0.002 + hazard_prox * 0.04 + doorway_prox * 0.12
            if random.random() < freeze_chance:
                self.is_freezing = True
                self.freeze_duration = random.randint(30, 90)

    def _get_hazard_proximity(self):
        """Returns 0-1 awareness score: rises 3.5 units before the obstacle surface.
        This ensures physiology responds BEFORE and DURING contact, not just when inside."""
        max_p = 0.0
        for h in self.obstacles:
            d = math.sqrt((self.pos["x"]-h["x"])**2 + (self.pos["z"]-h["z"])**2)
            awareness = h["radius"] + 3.5
            if d < awareness:
                max_p = max(max_p, (awareness - d) / awareness)
        return max_p

    def _apply_parkinson_motor(self):
        """Inject clinically-grounded Parkinson's motor disturbances each physics tick."""
        speed = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2)

        # --- Festination: involuntary forward acceleration (can't stop) ---
        if not self.festination_active:
            # Trigger ~once every 25-40 seconds; only while moving
            if not self.is_manual and speed > self.max_speed * 0.3 and random.random() < 0.0008:
                self.festination_active = True
                self.festination_timer  = random.randint(50, 120)
                self.festination_mult   = random.uniform(2.2, 3.5)
        else:
            self.festination_timer -= 1
            if self.festination_timer <= 0:
                self.festination_active = False
                self.festination_mult   = 1.0
                # Festination often ends in a near-fall / freeze
                self.is_freezing     = True
                self.freeze_duration = random.randint(25, 60)
                self.collision_stress_boost = min(0.9, self.collision_stress_boost + 0.5)

        # --- Gait irregularity: shuffling noise (irregular step rhythm) ---
        self.gait_phase += random.gauss(0.6, 0.15)   # irregular cadence
        shuffle_noise = math.sin(self.gait_phase) * 0.0004
        self.vel["x"] += shuffle_noise
        self.vel["z"] += shuffle_noise * random.uniform(-1, 1)

        # --- Postural drift: lateral instability (random sideways push) ---
        if speed > 0.005 and random.random() < 0.04:
            perp_x = -self.vel["z"]
            perp_z =  self.vel["x"]
            pmag = math.sqrt(perp_x**2 + perp_z**2)
            if pmag > 0:
                drift = random.gauss(0, 0.0006)
                self.vel["x"] += (perp_x / pmag) * drift
                self.vel["z"] += (perp_z / pmag) * drift

    def _get_obstacle_repulsion(self):
        """Steering-layer repulsion: pushes avatar away from obstacles before a hard collision."""
        rx, rz = 0.0, 0.0
        for obs in self.obstacles:
            dx = self.pos["x"] - obs["x"]
            dz = self.pos["z"] - obs["z"]
            dist = math.sqrt(dx * dx + dz * dz)
            influence = obs["radius"] + 2.5  # Start deflecting 2.5 units before surface
            if 0.01 < dist < influence:
                strength = ((influence - dist) / influence) ** 2  # Quadratic falloff
                rx += (dx / dist) * strength * self.max_force * 3.0
                rz += (dz / dist) * strength * self.max_force * 3.0
        return rx, rz

    def _handle_collisions(self):
        """Hard collision resolution: cage walls (reflect) + obstacle volumes (push-out + bounce)."""
        ch = self.cage_half
        ar = self.avatar_radius

        # --- Cage boundary reflection ---
        if self.pos["x"] < -ch + ar:
            self.pos["x"] = -ch + ar
            self.vel["x"] = abs(self.vel["x"]) * 0.25
        elif self.pos["x"] > ch - ar:
            self.pos["x"] = ch - ar
            self.vel["x"] = -abs(self.vel["x"]) * 0.25

        if self.pos["z"] < -ch + ar:
            self.pos["z"] = -ch + ar
            self.vel["z"] = abs(self.vel["z"]) * 0.25
        elif self.pos["z"] > ch - ar:
            self.pos["z"] = ch - ar
            self.vel["z"] = -abs(self.vel["z"]) * 0.25

        # --- Per-obstacle rigid circle collision ---
        for obs in self.obstacles:
            dx = self.pos["x"] - obs["x"]
            dz = self.pos["z"] - obs["z"]
            dist_sq = dx * dx + dz * dz
            min_dist = obs["radius"] + ar
            if dist_sq < min_dist * min_dist and dist_sq > 0.0001:
                dist = math.sqrt(dist_sq)
                nx = dx / dist   # unit normal: obstacle-centre → avatar
                nz = dz / dist
                # Separate: push avatar out of penetration zone
                self.pos["x"] += nx * (min_dist - dist)
                self.pos["z"] += nz * (min_dist - dist)
                # Damped velocity reflection (restitution 0.25)
                dot = self.vel["x"] * nx + self.vel["z"] * nz
                if dot < 0:   # Moving toward obstacle
                    self.vel["x"] -= 1.25 * dot * nx
                    self.vel["z"] -= 1.25 * dot * nz
                    # Cap post-bounce speed
                    spd = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2)
                    if spd > self.max_speed * 0.55:
                        scale = (self.max_speed * 0.55) / spd
                        self.vel["x"] *= scale
                        self.vel["z"] *= scale
                    # ← Acute physiological impact: inject stress spike
                    self.collision_stress_boost = min(0.90, self.collision_stress_boost + 0.35)

    def generate_reading(self):
        self.step += 1
        self._update_physics()
        self._update_physiology()
        
        breath = (math.sin(self.step * 0.1) + 1.0) * 0.5
        sway = math.sin(self.step * 0.05) * self.tremor_intensity
        
        # Activity level amplified for better visual feedback
        activity = math.sqrt(self.vel["x"]**2 + self.vel["z"]**2) * 2.5
        
        return {
            "heart_rate":        round(self.current_hr + (random.random()-0.5)*2.5, 1),
            "temperature":       round(self.current_temp + (random.random()-0.5)*0.1, 2),
            "activity_level":    round(activity, 2),
            "position":          {"x": round(self.pos["x"], 3), "z": round(self.pos["z"], 3)},
            "hazard_proximity":  round(self._get_hazard_proximity(), 2),
            "tremor_intensity":  round(self.tremor_intensity, 3),
            "is_freezing":       self.is_freezing,
            "festination_active": self.festination_active,
            "stress_level":      round(self.stress_level, 2),
            "breath_phase":      round(breath, 2),
            "sway":              round(sway, 2),
            "timestamp":         datetime.utcnow().isoformat()
        }

engine = DigitalTwinEngine()
