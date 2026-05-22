"""Local cost-grid + DWA-lite planner for the RoboVerse 2026 qualifier mission.

Architecture:
  - Drone-centric, world-aligned rolling cost grid (20m x 20m at 0.4m/cell).
  - Each tick: project depth image into the grid (obstacle evidence + free-space rays).
  - DWA-lite samples short forward trajectories; picks lowest-cost one biased toward goal.
  - Frontier selection: nearest known-free cell adjacent to an unknown cell.

The mission feeds in (depth_frame, drone_pose) and gets back (forward_speed, yaw_rate)
plus optional debug hooks. No keyboards, no hardcoded waypoints, no GPS.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


# ---------------- camera intrinsics (from CLAUDE.md) ----------------
FX = 433.0
FY = 433.0
CX_PX = 320.0
CY_PX = 240.0


@dataclass
class GridConfig:
    size_m: float = 20.0            # world-aligned window edge length
    cell_m: float = 0.4             # cell resolution
    inflate_cells: int = 1          # obstacle inflation radius (drone half-width)
    occ_decay: float = 0.92         # per-tick decay of obstacle evidence
    free_decay: float = 0.97        # per-tick decay of free evidence (slower → memory)
    occ_hit: float = 0.6            # obstacle evidence added per ray-hit
    free_hit: float = 0.25          # free evidence added per ray pass-through cell
    occ_known_thresh: float = 0.35  # occ above this → "occupied"
    free_known_thresh: float = 0.25 # free above this → "known free"


@dataclass
class DwaConfig:
    horizon_s: float = 1.4              # how far ahead we roll out
    step_s: float = 0.15                # simulation step inside rollout
    forward_speeds: tuple = (0.4, 0.9, 1.3)
    yaw_rates_deg: tuple = (-50.0, -30.0, -15.0, 0.0, 15.0, 30.0, 50.0)
    stationary_yaw_rates_deg: tuple = (-60.0, -40.0, -20.0, 20.0, 40.0, 60.0)
    hover_option: bool = True           # allow 0-velocity option as a fallback
    hard_cost_reject: float = 0.65      # any cell on path with occ >= this → reject (was 0.55 — too quick to abandon marginal paths)
    weight_goal: float = 6.0            # reward for closing distance to goal
    weight_max_cost: float = 4.0        # penalise highest-cost cell on the path
    weight_avg_cost: float = 2.0        # penalise average cost
    weight_yaw: float = 0.05            # mild bias against spinning
    weight_unknown: float = 1.5         # mild penalty for crossing unknown cells
    footprint_cells: int = 1            # check cost in (2*r+1) square around each rollout point
    weight_heading_align: float = 1.0   # bonus for turning toward goal when stuck
    hover_penalty: float = 0.6          # small constant penalty for v=0 w=0 to break ties
    yaw_commit_bonus: float = 0.5       # bonus for keeping last yaw-rate sign
    flip_penalty: float = 0.7           # penalty for reversing yaw direction (large — kill oscillation)
    stuck_commitment_boost: float = 1.5 # extra multiplier on commit/flip once stuck_count exceeds threshold
    stuck_frames_threshold: int = 8     # how many non-forward replans before we boost commitment


@dataclass
class FrontierConfig:
    visit_radius_m: float = 1.0          # mark cells within this radius "visited"
    reach_tolerance_m: float = 1.0       # frontier counts as reached at this distance
    max_search_radius_m: float = 12.0    # search horizon when picking a frontier
    min_distance_m: float = 3.0          # ignore frontiers closer than this — forces outward exploration
    timeout_s: float = 20.0              # abandon frontier after this without progress
    heading_bias_weight: float = 1.2     # prefer frontiers aligned with current heading (dominates distance for close picks)
    blacklist_radius_m: float = 1.2      # blacklist a neighbourhood, not a single cell


class CostGrid:
    """Drone-centric, world-aligned rolling occupancy / free / visited grid.

    Frames:
      - World frame: NED (X=north, Y=east). Stored grid is aligned with world axes
        so cells don't need to rotate when drone yaws.
      - Grid frame: cell indices (i, j) where i = (n - origin_n) / cell_m, similar for j.
        Origin is the lower-NE corner; recentered on the drone when it moves >1 cell.
    """

    def __init__(self, config: GridConfig | None = None):
        self.cfg = config or GridConfig()
        self.n_cells = int(self.cfg.size_m / self.cfg.cell_m)
        self.occ = np.zeros((self.n_cells, self.n_cells), dtype=np.float32)
        self.free = np.zeros((self.n_cells, self.n_cells), dtype=np.float32)
        self.visited = np.zeros((self.n_cells, self.n_cells), dtype=np.bool_)
        # World coords of grid cell (0,0)
        self.origin_n = 0.0
        self.origin_e = 0.0
        self._half_m = self.cfg.size_m / 2.0
        self._inited = False

    # ---- coordinate helpers ----
    def world_to_cell(self, n: float, e: float) -> tuple[int, int]:
        i = int((n - self.origin_n) / self.cfg.cell_m)
        j = int((e - self.origin_e) / self.cfg.cell_m)
        return i, j

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        n = self.origin_n + (i + 0.5) * self.cfg.cell_m
        e = self.origin_e + (j + 0.5) * self.cfg.cell_m
        return n, e

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.n_cells and 0 <= j < self.n_cells

    # ---- rolling window recentre ----
    def recenter(self, drone_n: float, drone_e: float) -> None:
        """Shift grid so the drone sits near the centre. Only shifts in whole cells."""
        if not self._inited:
            self.origin_n = drone_n - self._half_m
            self.origin_e = drone_e - self._half_m
            self._inited = True
            return

        # Desired origin such that drone is at the centre
        desired_origin_n = drone_n - self._half_m
        desired_origin_e = drone_e - self._half_m
        di = int(round((desired_origin_n - self.origin_n) / self.cfg.cell_m))
        dj = int(round((desired_origin_e - self.origin_e) / self.cfg.cell_m))
        if di == 0 and dj == 0:
            return

        # Roll arrays by (-di, -dj) — cells outside the new window get zeroed
        self.occ = self._roll(self.occ, -di, -dj, 0.0)
        self.free = self._roll(self.free, -di, -dj, 0.0)
        self.visited = self._roll(self.visited, -di, -dj, False)
        self.origin_n += di * self.cfg.cell_m
        self.origin_e += dj * self.cfg.cell_m

    @staticmethod
    def _roll(arr: np.ndarray, di: int, dj: int, fill):
        result = np.full_like(arr, fill)
        src_i_start = max(0, -di)
        src_i_end = arr.shape[0] - max(0, di)
        src_j_start = max(0, -dj)
        src_j_end = arr.shape[1] - max(0, dj)
        if src_i_start >= src_i_end or src_j_start >= src_j_end:
            return result
        dst_i_start = max(0, di)
        dst_i_end = dst_i_start + (src_i_end - src_i_start)
        dst_j_start = max(0, dj)
        dst_j_end = dst_j_start + (src_j_end - src_j_start)
        result[dst_i_start:dst_i_end, dst_j_start:dst_j_end] = \
            arr[src_i_start:src_i_end, src_j_start:src_j_end]
        return result

    # ---- per-tick decay + visit marking ----
    def tick_decay(self) -> None:
        self.occ *= self.cfg.occ_decay
        self.free *= self.cfg.free_decay

    def mark_visited(self, drone_n: float, drone_e: float, radius_m: float) -> None:
        i0, j0 = self.world_to_cell(drone_n, drone_e)
        r = int(math.ceil(radius_m / self.cfg.cell_m))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj > r * r:
                    continue
                i, j = i0 + di, j0 + dj
                if self.in_bounds(i, j):
                    self.visited[i, j] = True
                    self.free[i, j] = max(self.free[i, j], self.cfg.free_known_thresh + 0.1)

    # ---- depth integration ----
    def integrate_depth(
        self,
        depth: np.ndarray,
        drone_n: float,
        drone_e: float,
        yaw_deg: float,
        max_range_m: float = 10.0,
        n_columns: int = 64,
        row_band: tuple[float, float] = (0.35, 0.65),
    ) -> None:
        """Project depth image columns into the grid.

        For each sampled column: take the median depth in the central row band,
        compute the world-frame endpoint, then ray-march to mark free cells and
        increment the obstacle cell at the end.
        """
        h, w = depth.shape
        r0 = int(h * row_band[0])
        r1 = int(h * row_band[1])
        cols = np.linspace(0, w - 1, n_columns).astype(np.int32)

        yaw_rad = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        for col in cols:
            sample = depth[r0:r1, col]
            valid = sample[np.isfinite(sample) & (sample > 0.2)]
            if valid.size == 0:
                continue
            d = float(np.percentile(valid, 20))   # near-edge → conservative
            d = min(d, max_range_m)
            hit = d < max_range_m - 0.5          # treat clipped depths as "free clear out"

            # Camera frame: x_right = (col - cx)*d/fx ; z_forward = d
            x_cam = (col - CX_PX) * d / FX
            z_cam = d
            # Body frame (forward-right-down): forward = +z_cam, right = +x_cam
            forward = z_cam
            right = x_cam
            # World NED frame (yaw rotates body forward toward N at yaw=0):
            dn = cos_y * forward - sin_y * right
            de = sin_y * forward + cos_y * right
            end_n = drone_n + dn
            end_e = drone_e + de

            self._raycast(drone_n, drone_e, end_n, end_e, mark_hit=hit)

    def _raycast(self, n0: float, e0: float, n1: float, e1: float, mark_hit: bool) -> None:
        i0, j0 = self.world_to_cell(n0, e0)
        i1, j1 = self.world_to_cell(n1, e1)
        cells = _bresenham(i0, j0, i1, j1)
        if not cells:
            return
        # Free for everything along the ray except the final cell
        for (i, j) in cells[:-1]:
            if self.in_bounds(i, j):
                self.free[i, j] = min(1.0, self.free[i, j] + self.cfg.free_hit)
        # Endpoint: obstacle hit (with inflation) or free if cleared out
        last_i, last_j = cells[-1]
        if mark_hit:
            for di in range(-self.cfg.inflate_cells, self.cfg.inflate_cells + 1):
                for dj in range(-self.cfg.inflate_cells, self.cfg.inflate_cells + 1):
                    i, j = last_i + di, last_j + dj
                    if self.in_bounds(i, j):
                        weight = 1.0 if (di == 0 and dj == 0) else 0.5
                        self.occ[i, j] = min(1.0, self.occ[i, j] + self.cfg.occ_hit * weight)
        else:
            if self.in_bounds(last_i, last_j):
                self.free[last_i, last_j] = min(1.0, self.free[last_i, last_j] + self.cfg.free_hit)

    # ---- queries ----
    def cost_at(self, n: float, e: float) -> float:
        i, j = self.world_to_cell(n, e)
        if not self.in_bounds(i, j):
            return 1.0  # outside window → treat as unknown/risky
        return float(self.occ[i, j])

    def is_unknown(self, i: int, j: int) -> bool:
        if not self.in_bounds(i, j):
            return False
        return (self.occ[i, j] < self.cfg.occ_known_thresh
                and self.free[i, j] < self.cfg.free_known_thresh
                and not self.visited[i, j])

    def is_free(self, i: int, j: int) -> bool:
        if not self.in_bounds(i, j):
            return False
        return (self.occ[i, j] < self.cfg.occ_known_thresh
                and (self.free[i, j] >= self.cfg.free_known_thresh or self.visited[i, j]))

    def is_occupied(self, i: int, j: int) -> bool:
        if not self.in_bounds(i, j):
            return False
        return self.occ[i, j] >= self.cfg.occ_known_thresh


def _wrap_180(deg: float) -> float:
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def _bresenham(i0: int, j0: int, i1: int, j1: int) -> list[tuple[int, int]]:
    cells = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    i, j = i0, j0
    for _ in range(di + dj + 1):
        cells.append((i, j))
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj
    return cells


# ---------------- DWA-lite ----------------

@dataclass
class Command:
    forward_m_s: float
    yaw_rate_deg_s: float
    score: float
    rejected: bool = False
    rollout: list = None
    label: str = "MOVE"


class DwaPlanner:
    def __init__(self, grid: CostGrid, config: DwaConfig | None = None):
        self.grid = grid
        self.cfg = config or DwaConfig()
        self.last_yaw_rate = 0.0   # last commanded yaw rate — used for commitment bias
        self.stuck_count = 0       # number of consecutive non-forward replans

    def reset_commitment(self) -> None:
        """Clear yaw-direction memory — call when goal changes so we re-evaluate fresh."""
        self.last_yaw_rate = 0.0
        self.stuck_count = 0

    def plan(
        self,
        drone_n: float,
        drone_e: float,
        yaw_deg: float,
        goal_n: float | None,
        goal_e: float | None,
    ) -> Command:
        best = None
        candidates: list[tuple[float, float]] = []
        for v in self.cfg.forward_speeds:
            for w in self.cfg.yaw_rates_deg:
                candidates.append((v, w))
        for w in self.cfg.stationary_yaw_rates_deg:
            candidates.append((0.0, w))
        if self.cfg.hover_option:
            candidates.append((0.0, 0.0))

        # Boost commitment when stuck — once we've been turning >stuck_frames, dominate noise
        commit_mul = self.cfg.stuck_commitment_boost if self.stuck_count >= self.cfg.stuck_frames_threshold else 1.0

        for v, w in candidates:
            rollout = self._simulate(drone_n, drone_e, yaw_deg, v, w)
            cost = self._score(rollout, goal_n, goal_e, yaw_deg, v, w)
            if cost is None:
                continue
            if abs(self.last_yaw_rate) > 1.0:
                if w * self.last_yaw_rate > 0:
                    cost += self.cfg.yaw_commit_bonus * commit_mul
                elif w * self.last_yaw_rate < 0:
                    cost -= self.cfg.flip_penalty * commit_mul
            if best is None or cost > best[0]:
                best = (cost, v, w, rollout)

        if best is None:
            if abs(self.last_yaw_rate) > 1.0:
                yaw_dir = 1.0 if self.last_yaw_rate > 0 else -1.0
            else:
                yaw_dir = self._goal_yaw_dir(drone_n, drone_e, yaw_deg, goal_n, goal_e)
            chosen_w = 40.0 * yaw_dir
            self.last_yaw_rate = chosen_w
            self.stuck_count += 1
            return Command(
                forward_m_s=0.0,
                yaw_rate_deg_s=chosen_w,
                score=-1.0,
                rejected=True,
                rollout=[(drone_n, drone_e)],
                label="ROTATE_SAFE",
            )

        score, v, w, rollout = best
        self.last_yaw_rate = w
        if v >= 0.4:
            self.stuck_count = 0  # making forward progress
        else:
            self.stuck_count += 1
        if v > 0 and abs(w) < 5:
            label = "FORWARD"
        elif v == 0 and abs(w) < 5:
            label = "HOVER"
        elif v == 0:
            label = "TURN"
        else:
            label = "ARC"
        return Command(forward_m_s=v, yaw_rate_deg_s=w, score=score, rollout=rollout, label=label)

    def _simulate(self, n0: float, e0: float, yaw_deg: float, v: float, w: float):
        steps = int(self.cfg.horizon_s / self.cfg.step_s)
        n, e, yaw = n0, e0, yaw_deg
        pts = []
        for _ in range(steps):
            yaw_rad = math.radians(yaw)
            n += v * math.cos(yaw_rad) * self.cfg.step_s
            e += v * math.sin(yaw_rad) * self.cfg.step_s
            yaw += w * self.cfg.step_s
            pts.append((n, e))
        return pts

    def _score(self, rollout, goal_n, goal_e, yaw_deg, v, w):
        max_cost = 0.0
        sum_cost = 0.0
        unknown_count = 0
        r = self.cfg.footprint_cells
        for (n, e) in rollout:
            i0, j0 = self.grid.world_to_cell(n, e)
            if not self.grid.in_bounds(i0, j0):
                return None
            # Sample the drone footprint (small box) — catches near-misses
            local_max = 0.0
            local_sum = 0.0
            samples = 0
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if not self.grid.in_bounds(i, j):
                        continue
                    c = float(self.grid.occ[i, j])
                    if c >= self.cfg.hard_cost_reject:
                        return None
                    local_max = max(local_max, c)
                    local_sum += c
                    samples += 1
            samples = max(1, samples)
            max_cost = max(max_cost, local_max)
            sum_cost += local_sum / samples
            if self.grid.is_unknown(i0, j0):
                unknown_count += 1
        avg_cost = sum_cost / max(1, len(rollout))

        # Goal-progress: distance reduction from start to end of rollout.
        # Also: when we can't move forward usefully, turning to face the goal counts as progress.
        heading_align = 0.0
        if goal_n is not None and goal_e is not None:
            start = rollout[0] if rollout else (0.0, 0.0)
            end = rollout[-1]
            d_start = math.hypot(goal_n - start[0], goal_e - start[1])
            d_end = math.hypot(goal_n - end[0], goal_e - end[1])
            progress = d_start - d_end

            # Heading alignment progress: |yaw_err_start| - |yaw_err_end|
            yaw_start = yaw_deg
            yaw_end = yaw_deg + w * self.cfg.horizon_s
            target_yaw = math.degrees(math.atan2(goal_e - start[1], goal_n - start[0]))
            err_start = _wrap_180(target_yaw - yaw_start)
            err_end = _wrap_180(target_yaw - yaw_end)
            heading_align = (abs(err_start) - abs(err_end)) / 90.0  # normalised
        else:
            progress = v * self.cfg.horizon_s

        unknown_frac = unknown_count / max(1, len(rollout))
        yaw_effort = abs(w) / 50.0

        # Hover penalty so we don't sit still when something's blocking us
        is_hover = v == 0.0 and abs(w) < 1e-6
        hover_pen = self.cfg.hover_penalty if is_hover else 0.0

        return (self.cfg.weight_goal * progress
                + self.cfg.weight_heading_align * heading_align
                - self.cfg.weight_max_cost * max_cost
                - self.cfg.weight_avg_cost * avg_cost
                - self.cfg.weight_unknown * unknown_frac
                - self.cfg.weight_yaw * yaw_effort
                - hover_pen)

    @staticmethod
    def _goal_yaw_dir(n, e, yaw_deg, gn, ge):
        if gn is None or ge is None:
            return 1.0
        target = math.degrees(math.atan2(ge - e, gn - n))
        err = target - yaw_deg
        while err > 180: err -= 360
        while err < -180: err += 360
        return 1.0 if err > 0 else -1.0


# ---------------- frontier picker ----------------

class FrontierPicker:
    def __init__(self, grid: CostGrid, config: FrontierConfig | None = None):
        self.grid = grid
        self.cfg = config or FrontierConfig()
        self.blacklist: set[tuple[int, int]] = set()  # cells abandoned recently

    def blacklist_cell(self, n: float, e: float) -> None:
        """Blacklist a neighbourhood, not just one cell — so we don't immediately re-pick a cell 0.4m away."""
        i0, j0 = self.grid.world_to_cell(n, e)
        r = int(math.ceil(self.cfg.blacklist_radius_m / self.grid.cfg.cell_m))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= r * r:
                    self.blacklist.add((i0 + di, j0 + dj))

    def clear_blacklist(self) -> None:
        self.blacklist.clear()

    def pick(self, drone_n: float, drone_e: float, yaw_deg: float | None = None,
             prefer_far: bool = False) -> tuple[float, float] | None:
        """Return (n, e) of an outward exploration target.

        Strategy:
          - Look at frontier cells (free, adjacent to unknown, unvisited).
          - REJECT any closer than min_distance_m — forces the drone to commit.
          - Score = (-distance OR +distance if prefer_far) + heading_bias * alignment.
          - prefer_far=True is escape mode: pick the furthest viable target.
          - Fallback: any unknown cell beyond min_distance.
        """
        i0, j0 = self.grid.world_to_cell(drone_n, drone_e)
        max_cells = int(self.cfg.max_search_radius_m / self.grid.cfg.cell_m)
        min_cells = self.cfg.min_distance_m / self.grid.cfg.cell_m

        cos_y = math.cos(math.radians(yaw_deg)) if yaw_deg is not None else None
        sin_y = math.sin(math.radians(yaw_deg)) if yaw_deg is not None else None

        best_frontier = None
        best_unknown = None
        best_frontier_score = -float("inf")
        best_unknown_score = -float("inf")

        for di in range(-max_cells, max_cells + 1):
            for dj in range(-max_cells, max_cells + 1):
                i, j = i0 + di, j0 + dj
                if not self.grid.in_bounds(i, j):
                    continue
                if (i, j) in self.blacklist:
                    continue
                if self.grid.is_occupied(i, j):
                    continue
                d = math.hypot(di, dj)
                if d > max_cells or d < min_cells:
                    continue

                # alignment with current heading: dot(unit_vec_to_cell, heading_unit_vec)
                align = 0.0
                if cos_y is not None:
                    align = (di * cos_y + dj * sin_y) / max(d, 1e-6)

                # Distance term: negative (prefer near) or positive (prefer far / escape mode).
                # Align in [-1, 1]; +1 = straight ahead.
                dist_term = d / max_cells if prefer_far else -d / max_cells
                score = dist_term + self.cfg.heading_bias_weight * align

                if self.grid.is_free(i, j) and not self.grid.visited[i, j]:
                    if self._has_unknown_neighbour(i, j) and score > best_frontier_score:
                        best_frontier = (i, j)
                        best_frontier_score = score
                elif self.grid.is_unknown(i, j) and score > best_unknown_score:
                    best_unknown = (i, j)
                    best_unknown_score = score

        chosen = best_frontier or best_unknown
        if chosen is None:
            return None
        return self.grid.cell_to_world(*chosen)

    def _has_unknown_neighbour(self, i: int, j: int) -> bool:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if self.grid.is_unknown(i + di, j + dj):
                    return True
        return False
