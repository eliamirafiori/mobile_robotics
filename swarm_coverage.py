#!/usr/bin/env python3
"""
Decentralised, failure-adaptive area coverage for a drone swarm (2-D grid simulator).

There is NO central coordinator. Every drone runs the same on-board loop:

  1. SENSE        sweep the cells under its (square) camera footprint and update its own
                  belief map (UNKNOWN / COVERED / BLOCKED). No-fly cells are only learned
                  when a drone gets close enough to see them.
  2. COMMUNICATE  with drones inside radio range: merge belief maps (cell-wise max) and a
                  "peer table" (last known position / capability / timestamp of each drone).
                  Information therefore spreads by gossip across the swarm, multi-hop.
  3. PARTITION    using only its own peer table, split the still-uncovered cells between the
                  peers it believes are alive with a capability-weighted Voronoi partition
                  (distance / sqrt(capability)). A peer whose timestamp is older than the
                  heartbeat timeout is treated as lost, so its territory is automatically
                  taken over by whoever is closest -- this is the adaptation mechanism.
  4. ACT          inside its own territory, pick the viewpoint with the best
                  information-gain / path-cost ratio (frontier-style utility), plan a path
                  with BFS on its own belief map, and fly it.

Events (failure, motor/sensor degradation, comm loss, new no-fly zone, reinforcement drone)
are injected into the "world" only; drones find out through sensing and gossip.

Usage:
    python swarm_coverage.py                       # adaptive swarm, saves coverage_result.png
    python swarm_coverage.py --compare             # adaptive vs. static partition baseline
    python swarm_coverage.py --gif swarm.gif       # also write an animation
    python swarm_coverage.py --no-events           # same mission, nothing goes wrong
    python swarm_coverage.py --drones 8 --grid 80 --comm-range 15 --seed 3

Requires: numpy, matplotlib
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

UNKNOWN, COVERED, BLOCKED = 0, 1, 2  # belief states; UNKNOWN == "still needs coverage"


# --------------------------------------------------------------------------- config / events
@dataclass
class Config:
    grid: int = 50                 # side of the square area, in cells
    n_drones: int = 5
    sensor_radius: int = 2         # camera footprint half-width (2 -> 5x5 cells)
    speed: float = 1.0             # cells per step
    comm_range: float = 25.0       # radio range, in cells
    battery: float = 600.0         # flight range, in cells
    heartbeat_timeout: int = 25    # steps without news before a peer is considered lost
    replan_period: int = 6         # steps between forced re-planning (picks up gossip changes)
    target_coverage: float = 0.99
    max_steps: int = 1500
    seed: int = 0
    adaptive: bool = True          # False -> static partition computed once at launch
    frame_every: int = 0           # >0 -> keep animation frames


@dataclass
class Event:
    t: int
    kind: str            # FAIL | SLOW | SENSOR | COMM | NOFLY | JOIN
    drone: int = -1
    value: float = 0.0
    rect: tuple = ()     # NOFLY: (row, col, height, width)
    label: str = ""


def default_events(cfg: Config) -> list[Event]:
    n = cfg.grid
    return [
        Event(30, "FAIL", drone=1, label="drone 1 crashes"),
        Event(50, "SLOW", drone=3, value=0.4, label="drone 3 motor degraded (40% speed)"),
        Event(70, "SENSOR", drone=0, value=1, label="drone 0 camera degraded (3x3 footprint)"),
        Event(90, "NOFLY", rect=(int(.30 * n), int(.35 * n), int(.15 * n), int(.25 * n)),
              label="new no-fly zone"),
        Event(110, "JOIN", label="reinforcement drone launched"),
        Event(130, "COMM", drone=2, value=0.15, label="drone 2 radio degraded (15% range)"),
    ]


@dataclass(frozen=True)
class PeerInfo:
    pos: tuple
    weight: float
    stamp: int


# --------------------------------------------------------------------------- helpers
def box_sum(mask: np.ndarray, r: int) -> np.ndarray:
    """Sum of `mask` over the (2r+1)x(2r+1) window centred on every cell (integral image)."""
    k = 2 * r + 1
    H, W = mask.shape
    P = np.pad(mask.astype(np.int32), r)
    cs = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.int32)
    cs[1:, 1:] = P.cumsum(0).cumsum(1)
    return cs[k:k + H, k:k + W] - cs[0:H, k:k + W] - cs[k:k + H, 0:W] + cs[0:H, 0:W]


# --------------------------------------------------------------------------- the drone "brain"
class Drone:
    def __init__(self, uid: int, pos: tuple, cfg: Config):
        self.id = uid
        self.pos = pos                                  # (row, col)
        self.speed = cfg.speed
        self.radius = cfg.sensor_radius
        self.comm_range = cfg.comm_range
        self.battery0 = self.battery = cfg.battery
        self.alive = True
        self.belief = np.zeros((cfg.grid, cfg.grid), np.int8)
        self.peers: dict[int, PeerInfo] = {}
        self.frozen: list = []                          # only used by the static baseline
        self.target = None
        self.path: deque = deque()
        self.budget = 0.0
        self.replan = True
        self.last_plan = -10 ** 9
        self.share = 0.0                                # my fraction of the remaining area
        self.distance = 0.0
        self.swept = np.zeros((cfg.grid, cfg.grid), bool)   # cells I have swept myself

    @property
    def weight(self) -> float:
        """Capability: speed x swath width x remaining energy."""
        return max(self.speed, 0.05) * (2 * self.radius + 1) * max(self.battery / self.battery0, 0.05)

    # ---- peer table / heartbeat
    def heartbeat(self, t: int):
        self.peers[self.id] = PeerInfo(self.pos, self.weight, t)

    def anchors(self, world: "World") -> list:
        if not world.cfg.adaptive:
            return self.frozen
        out = []
        for pid, info in self.peers.items():
            if pid == self.id:
                out.append((pid, self.pos, self.weight))
            elif world.t - info.stamp <= world.cfg.heartbeat_timeout:
                out.append((pid, info.pos, info.weight))
        return out

    # ---- planning
    def _bfs(self):
        H, W = self.belief.shape
        W2 = W + 2
        blk = np.ones((H + 2, W2), bool)
        blk[1:-1, 1:-1] = self.belief == BLOCKED
        blk = blk.ravel().tolist()
        dist = [-1] * len(blk)
        parent = [-1] * len(blk)
        s = (self.pos[0] + 1) * W2 + self.pos[1] + 1
        dist[s] = 0
        q = deque([s])
        offs = (-W2 - 1, -W2, -W2 + 1, -1, 1, W2 - 1, W2, W2 + 1)
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for o in offs:
                v = u + o
                if dist[v] < 0 and not blk[v]:
                    dist[v] = du
                    parent[v] = u
                    q.append(v)
        return np.array(dist).reshape(H + 2, W2)[1:-1, 1:-1], parent, s, W2

    def plan(self, world: "World"):
        self.last_plan = world.t
        self.replan = False
        self.target = None
        self.path.clear()
        need = self.belief == UNKNOWN
        ys, xs = np.nonzero(need)
        anchors = self.anchors(world)
        ids = [a[0] for a in anchors]
        if len(ys) == 0 or self.id not in ids:
            self.share = 0.0
            return

        # 1) decentralised, capability-weighted Voronoi partition of the uncovered cells
        pts = np.stack([ys, xs], 1).astype(float)
        apos = np.array([a[1] for a in anchors], float)
        w = np.sqrt(np.array([a[2] for a in anchors], float))
        d = np.linalg.norm(pts[None] - apos[:, None], axis=2) / w[:, None]
        mine = d.argmin(0) == ids.index(self.id)
        self.share = float(mine.mean())

        # 2) utility / cost target selection inside my territory
        py, px = self.pos
        has_block = bool((self.belief == BLOCKED).any())
        if has_block:                                # path-aware distances around no-fly cells
            dist, parent, s, W2 = self._bfs()
            dcell = dist[ys, xs]
        else:                                        # open field: Chebyshev == 8-connected BFS
            dcell = np.maximum(np.abs(ys - py), np.abs(xs - px))
        reach = dcell >= 0
        gain = box_sum(need, self.radius)[ys, xs]
        score = gain / (dcell + 1 + self.radius)
        cand = reach & mine
        if not cand.any() and world.cfg.adaptive:   # territory finished -> help where work is left
            cand = reach
        if not cand.any():
            return                                   # nothing reachable (e.g. enclosed by no-fly cells)
        score = np.where(cand, score + 1e-6 * world.rng.random(len(ys)), -1.0)
        k = int(score.argmax())
        self.target = (int(ys[k]), int(xs[k]))

        # 3) path: straight line if clear (clean sweeps), otherwise BFS around no-fly cells
        line = self._line(self.target)
        if has_block and any(self.belief[c] == BLOCKED for c in line):
            v = (self.target[0] + 1) * W2 + self.target[1] + 1
            line = []
            while v != s:
                line.append((v // W2 - 1, v % W2 - 1))
                v = parent[v]
            line.reverse()
        self.path = deque(line)

    def _line(self, tgt) -> list:
        n = max(abs(tgt[0] - self.pos[0]), abs(tgt[1] - self.pos[1]))
        if n == 0:
            return []
        ys = np.rint(np.linspace(self.pos[0], tgt[0], n + 1)).astype(int)[1:]
        xs = np.rint(np.linspace(self.pos[1], tgt[1], n + 1)).astype(int)[1:]
        return list(zip(ys.tolist(), xs.tolist()))

    # ---- motion
    def move(self, world: "World"):
        if not self.path:
            self.budget = 0.0
            return
        self.budget += self.speed
        while self.budget >= 1 and self.path:
            nxt = self.path[0]
            if self.belief[nxt] == BLOCKED:      # discovered a no-fly cell on my path
                self.path.clear()
                self.target = None
                self.replan = True
                return
            step = math.hypot(nxt[0] - self.pos[0], nxt[1] - self.pos[1])
            self.pos = nxt
            self.path.popleft()
            self.budget -= 1
            self.battery -= step
            self.distance += step
            world.sense(self)
            if self.battery <= 0:
                self.alive = False
                world.log(f"drone {self.id} ran out of battery")
                return
        if not self.path:
            self.replan = True


# --------------------------------------------------------------------------- the world
class World:
    def __init__(self, cfg: Config, events: list[Event]):
        self.cfg = cfg
        self.events = sorted(events, key=lambda e: e.t)
        self.rng = np.random.default_rng(cfg.seed)
        n = cfg.grid
        self.covered = np.zeros((n, n), bool)
        self.blocked = np.zeros((n, n), bool)
        self.t = 0
        self.obs_redundant = 0
        self.idle_steps = 0
        cols = np.linspace(2, n - 3, cfg.n_drones).round().astype(int)
        self.drones = [Drone(i, (n - 1, int(c)), cfg) for i, c in enumerate(cols)]
        initial = {d.id: PeerInfo(d.pos, d.weight, 0) for d in self.drones}
        for d in self.drones:                      # everybody is briefed at launch
            d.peers = dict(initial)
            d.frozen = [(i, p.pos, p.weight) for i, p in initial.items()]
        self.hist_cov, self.hist_alive = [], []
        self.hist_share = defaultdict(list)        # id -> [(t, share)]
        self.traj = defaultdict(list)              # id -> [pos]
        self.fired: list[Event] = []
        self.frames = []
        for d in self.drones:                      # initial sweep
            self.sense(d)

    def log(self, msg: str):
        print(f"[t={self.t:4d}] {msg}")

    # ---- sensing (truth + the drone's own belief)
    def sense(self, d: Drone):
        H, W = self.covered.shape
        y, x = d.pos
        R = d.radius + 2                           # no-fly look-ahead
        win = (slice(max(y - R, 0), min(y + R + 1, H)), slice(max(x - R, 0), min(x + R + 1, W)))
        d.belief[win][self.blocked[win]] = BLOCKED
        r = d.radius
        win = (slice(max(y - r, 0), min(y + r + 1, H)), slice(max(x - r, 0), min(x + r + 1, W)))
        free = ~self.blocked[win]
        cov = self.covered[win]
        mine = d.swept[win]
        self.obs_redundant += int((free & cov & ~mine).sum())   # already done by someone else
        mine |= free
        cov |= free
        bw = d.belief[win]
        bw[free & (bw == UNKNOWN)] = COVERED

    # ---- events
    def nearest_free(self, pos):
        H, W = self.blocked.shape
        for rad in range(1, max(H, W)):
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if max(abs(dy), abs(dx)) != rad:
                        continue
                    y, x = pos[0] + dy, pos[1] + dx
                    if 0 <= y < H and 0 <= x < W and not self.blocked[y, x]:
                        return (y, x)
        return pos

    def apply(self, ev: Event):
        drone = next((d for d in self.drones if d.id == ev.drone and d.alive), None)
        needs_drone = ev.kind in ("FAIL", "SLOW", "SENSOR", "COMM")
        if needs_drone and drone is None:
            return
        self.log(f"EVENT {ev.kind}: {ev.label}")
        if ev.kind == "FAIL":
            drone.alive = False
        elif ev.kind == "SLOW":
            drone.speed = ev.value
        elif ev.kind == "SENSOR":
            drone.radius = int(ev.value)
        elif ev.kind == "COMM":
            drone.comm_range *= ev.value
        elif ev.kind == "NOFLY":
            r0, c0, h, w = ev.rect
            self.blocked[r0:r0 + h, c0:c0 + w] = True
            for d in self.drones:
                if d.alive and self.blocked[d.pos]:    # emergency sidestep
                    d.pos = self.nearest_free(d.pos)
                    d.path.clear()
                    d.replan = True
        elif ev.kind == "JOIN":
            if not self.cfg.adaptive:
                self.log("  (static baseline cannot absorb a new drone - ignored)")
                return
            n = self.cfg.grid
            nd = Drone(max(d.id for d in self.drones) + 1, (n - 1, 0), self.cfg)
            nd.peers = {nd.id: PeerInfo(nd.pos, nd.weight, self.t)}   # knows nobody yet
            self.drones.append(nd)
            self.sense(nd)
        self.fired.append(dataclasses.replace(ev))

    # ---- gossip
    def communicate(self, alive: list[Drone]):
        snap = {d.id: (d.belief.copy(), dict(d.peers)) for d in alive}
        for a in alive:
            for b in alive:
                if a is b or math.dist(a.pos, b.pos) > min(a.comm_range, b.comm_range):
                    continue
                belief, peers = snap[b.id]
                np.maximum(a.belief, belief, out=a.belief)
                for pid, info in peers.items():
                    cur = a.peers.get(pid)
                    if cur is None or info.stamp > cur.stamp:
                        a.peers[pid] = info

    # ---- main loop
    def coverage(self) -> float:
        free = ~self.blocked
        return float((self.covered & free).sum() / free.sum())

    def step(self):
        for ev in [e for e in self.events if e.t == self.t]:
            self.apply(ev)
        for d in [d for d in self.drones if d.alive]:
            stale = (d.replan or self.t - d.last_plan >= self.cfg.replan_period
                     or (d.target is not None and d.belief[d.target] != UNKNOWN))
            if stale:
                d.plan(self)
            d.move(self)
        alive = [d for d in self.drones if d.alive]
        for d in alive:
            d.heartbeat(self.t)
        self.communicate(alive)

        self.hist_cov.append(self.coverage())
        self.hist_alive.append(len(alive))
        for d in alive:
            self.hist_share[d.id].append((self.t, d.share))
            self.traj[d.id].append(d.pos)
        self.idle_steps = self.idle_steps + 1 if alive and all(d.target is None for d in alive) else 0
        if self.cfg.frame_every and self.t % self.cfg.frame_every == 0:
            self.frames.append((self.covered.copy(), self.blocked.copy(), self.t,
                                [(d.id, d.pos, d.alive) for d in self.drones], self.hist_cov[-1]))
        self.t += 1

    def finished(self) -> bool:
        return (self.t >= self.cfg.max_steps or not any(d.alive for d in self.drones)
                or (self.hist_cov and self.hist_cov[-1] >= self.cfg.target_coverage)
                or self.idle_steps >= 30)

    def summary(self) -> dict:
        return dict(mode="adaptive" if self.cfg.adaptive else "static", steps=self.t,
                    coverage=self.hist_cov[-1] if self.hist_cov else 0.0,
                    distance=sum(d.distance for d in self.drones),
                    overlap=self.obs_redundant / max(int(self.covered.sum()), 1),
                    alive=sum(d.alive for d in self.drones), total=len(self.drones))


def run(cfg: Config, events: list[Event]) -> World:
    world = World(cfg, events)
    while not world.finished():
        world.step()
    return world


# --------------------------------------------------------------------------- output
def print_summary(rows: list[dict]):
    print("\n mode      steps  coverage  total flight  inter-drone overlap  drones alive")
    for r in rows:
        print(f" {r['mode']:<9} {r['steps']:5d}  {r['coverage']:7.1%}  {r['distance']:12.0f}"
              f"  {r['overlap']:19.1%}  {r['alive']:6d}/{r['total']}")


def plot(world: World, baseline: World | None, out: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    state = np.where(world.blocked, 2, np.where(world.covered, 1, 0))
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    ax[0].imshow(state, cmap=ListedColormap(["#eeeeee", "#cfe8cf", "#222222"]), vmin=0, vmax=2)
    for d in world.drones:
        tr = np.array(world.traj[d.id])
        if len(tr):
            ax[0].plot(tr[:, 1], tr[:, 0], lw=1, color=colors(d.id % 10), label=f"drone {d.id}")
            ax[0].scatter(tr[-1, 1], tr[-1, 0], marker="o" if d.alive else "X", s=70,
                          color=colors(d.id % 10), edgecolor="k", zorder=3)
    ax[0].set_title(f"Final map ({world.coverage():.1%} covered)\n● flying   ✕ lost")
    ax[0].legend(fontsize=7, loc="upper right")

    ax[1].plot(world.hist_cov, label="adaptive" if world.cfg.adaptive else "static")
    if baseline is not None:
        ax[1].plot(baseline.hist_cov, "--",
                   label="adaptive" if baseline.cfg.adaptive else "static")
    for i, ev in enumerate(world.fired):
        ax[1].axvline(ev.t, color="grey", lw=.8, ls=":")
        ax[1].text(ev.t, 0.02 + 0.07 * (i % 4), f" {ev.kind}", fontsize=7, rotation=90, va="bottom")
    ax[1].set_ylim(0, 1.02)
    ax[1].set(xlabel="step", ylabel="coverage of free area", title="Coverage over time")
    ax[1].legend()
    ax[1].grid(alpha=.3)

    for pid, series in world.hist_share.items():
        ts, sh = zip(*series)
        sh = np.convolve(np.pad(sh, (4, 0), mode="edge"), np.ones(5) / 5, mode="valid")  # 5-step average
        ax[2].plot(ts, sh, color=colors(pid % 10), label=f"drone {pid}")
    for ev in world.fired:
        ax[2].axvline(ev.t, color="grey", lw=.8, ls=":")
    ax[2].set(xlabel="step", ylabel="share of remaining area (own view)",
              title="Decentralised territory re-allocation")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def save_gif(world: World, path: str):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    cmap = ListedColormap(["#eeeeee", "#cfe8cf", "#222222"])
    img = ax.imshow(np.zeros_like(world.covered, dtype=int), cmap=cmap, vmin=0, vmax=2)
    sc = ax.scatter([], [], s=60, edgecolor="k", zorder=3)
    ax.set_xticks([]), ax.set_yticks([])
    tab = plt.get_cmap("tab10")

    def update(i):
        cov, blk, t, drones, c = world.frames[i]
        img.set_data(np.where(blk, 2, np.where(cov, 1, 0)))
        sc.set_offsets([[p[1], p[0]] for _, p, _ in drones])
        sc.set_color([tab(u % 10) if a else "red" for u, _, a in drones])
        ax.set_title(f"t={t}   coverage {c:.1%}")
        return img, sc

    FuncAnimation(fig, update, frames=len(world.frames)).save(path, writer=PillowWriter(fps=10))
    print(f"saved {path}")


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--drones", type=int, default=5)
    ap.add_argument("--sensor-radius", type=int, default=2)
    ap.add_argument("--comm-range", type=float, default=25.0)
    ap.add_argument("--timeout", type=int, default=25, help="heartbeat timeout (steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--static", action="store_true", help="static partition baseline instead of adaptive")
    ap.add_argument("--compare", action="store_true", help="also run the other mode and compare")
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--out", default="coverage_result.png")
    ap.add_argument("--gif", default="", help="write an animation to this path")
    ap.add_argument("--show", action="store_true", help="open a window (needs a display)")
    a = ap.parse_args()

    import matplotlib
    if not a.show:
        matplotlib.use("Agg")

    cfg = Config(grid=a.grid, n_drones=a.drones, sensor_radius=a.sensor_radius,
                 comm_range=a.comm_range, heartbeat_timeout=a.timeout, seed=a.seed,
                 max_steps=a.max_steps, adaptive=not a.static, frame_every=5 if a.gif else 0)
    events = [] if a.no_events else default_events(cfg)

    world = run(cfg, events)
    rows = [world.summary()]
    other = None
    if a.compare:
        print("\n--- running the other mode on the identical scenario ---")
        other = run(dataclasses.replace(cfg, adaptive=not cfg.adaptive, frame_every=0), events)
        rows.append(other.summary())
    print_summary(rows)

    plot(world, other, a.out)
    if a.gif:
        save_gif(world, a.gif)
    if a.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
