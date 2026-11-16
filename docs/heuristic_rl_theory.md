In robotics and autonomous systems research, problems are intentionally decoupled into a hierarchical stack:

1. **Low-Level Control & Dynamics** (motor mixing, PID/MPC attitude stabilization, rigid-body physics)
2. **Perception & State Estimation** (SLAM, visual odometry, depth reconstruction)
3. **High-Level Decision-Making & Coordination** (area partitioning, task allocation, multi-agent RL)

As a computer engineer entering swarm coordination research, you should treat MuJoCo and the PD flight controller as a **black-box physics engine**—just like a software engineer treats the OS kernel or GPU driver when writing a distributed algorithm. Your research contribution lives in **Layer 3**: designing how agents make decisions under partial observability, limited communication, and sudden node failures.

The only detail you need to keep in mind from the lower layers is their **physical interface constraints**: drones have inertia (so changing target waypoints too rapidly causes physical oscillation) and cameras have finite fields of view with line-of-sight occlusion (which is why testing in MuJoCo is much more credible for a paper than testing on a simplified 2D grid).

---

## 1. How the Coverage Algorithms Work

All decentralized coverage algorithms solve the same distributed systems problem: **how can $N$ agents partition a shared spatial workload without a central server, while remaining fault-tolerant?**

### A. Dynamic Voronoi Partitioning + Hybrid Sweep (`voronoi`)

* **The Math:** Given the positions $p_1, \dots, p_N$ of active drones within communication range, drone $i$ defines its Voronoi cell $V_i$ as the set of all points $q$ in the arena closer to itself than to any active neighbor $j$:

$$V_i = \{ q \in \mathcal{E} \mid \Vert{}q - p_i\Vert{}_2 \le \Vert{}q - p_j\Vert{}_2, \; \forall j \in \mathcal{N}_i \}$$



In classical **Lloyd's Algorithm**, each drone computes the uncertainty-weighted center of mass (centroid) $C_i$ of its cell and flies there:

$$C_i = \frac{\sum_{q \in V_i} q \cdot u(q)}{\sum_{q \in V_i} u(q)}$$



where $u(q)$ is the uncertainty at cell $q$.
* **Why Pure Lloyd's Fails for Sweeping (and Our Fix):** Lloyd's algorithm is mathematically proven to converge to a *static* optimal deployment (ideal for positioning cell towers or stationary security cameras). However, if you want drones to *sweep* an area clean, symmetric unvisited edges pull equally in opposite directions, trapping the drone at the center. Blending **30% Voronoi centroid** (global territory balance) with **70% local greedy sweep** (highest $u(q)^2 / \text{distance}$) breaks that symmetry so the drone actively clears corners while staying inside its partition.
* **Implicit Coordination & Fault Tolerance:** Drones only need to broadcast their current `(x, y)` position. If Drone 0 dies, its position disappears from $\mathcal{N}_i$, and the geometric bisectors instantly expand to absorb Drone 0's territory without any explicit reassignment protocol.

### B. Cooperative Information-Gain Frontier Exploration (`frontier`)

* **The Math:** Instead of drawing rigid geometric boundaries, each drone scores candidate unvisited points $c$ using a utility function that balances **reward (information gain)** against **travel cost** and **peer overlap**:

$$U(c) = \alpha \underbrace{\sum_{\Vert{}q - c\Vert{} \le r_{\text{cam}}} u(q)}_{\text{Information Gain}} - \beta \Vert{}c - p_i\Vert{}_2 - \gamma P_{\text{peers}}(c)$$



where $P_{\text{peers}}(c)$ adds a penalty if a neighbor is closer to $c$ or has already declared a target near $c$.
* **Explicit Coordination Trade-Off:** Unlike Voronoi, Frontier works much better in non-convex spaces with buildings (`urban` / `ruins`) because it evaluates pockets of unexplored space rather than straight-line Euclidean bisectors. The trade-off is higher communication overhead: drones must broadcast both their position *and* their intended `target` waypoint so teammates don't chase the same frontier.

### C. Structured vs. Myopic Baselines (`spiral_inward`, `lawnmower`, `nearest_unvisited`)

* **Deterministic Patterns (`lawnmower`, `spiral_inward`):** Divide the geometry mathematically (horizontal bands or concentric rings). They achieve optimal path efficiency with zero overlap in empty convex spaces, but degrade around complex 3D obstacles.
* **Myopic Greedy (`nearest_unvisited`):** Minimizes only immediate step cost $\arg\min \Vert{}q - p_i\Vert{}$. Because it has no global horizon or spatial partitioning, it easily gets trapped in local minima or suffers from tie-breaking oscillations on symmetric grids.

---

## 2. How Reinforcement Learning (RL) Actually Works

In heuristic algorithms like Voronoi or Frontier, **you manually write the utility equations** and hand-tune weights ($\alpha, \beta, \gamma$). In RL, you replace those hand-crafted equations with a neural network $\pi_\theta(a \mid o)$ and let the agents discover the optimal decision policy through trial and error.

### Step 1: Formulating the Swarm as a Dec-POMDP

Because each drone only sees its local sensor map and nearby neighbors, the problem is modeled as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**:

* **Observation ($o_i \in \mathbb{R}^{110}$):** What drone $i$ knows locally at time $t$—its own normalized position/velocity, relative `(dx, dy)` vectors to its 3 closest active neighbors, and its local $10 \times 10$ uncertainty grid.
* **Action ($a_i \in [-1, 1]^2$):** The high-level 2D waypoint command `(target_x, target_y)` sent to the flight controller.
* **Reward ($r_t$):** A scalar feedback signal measuring how good the outcome was:

$$r_t = +0.15(\text{new cells seen by } i) + 0.05(\text{new cells seen by swarm}) - 0.30(\text{crowding}) - 0.02(\text{time penalty})$$



### Step 2: How the Neural Network Learns (PPO)

In `rl_swarm_env.py`, we use **PPO (Proximal Policy Optimization)**, which is an **Actor-Critic** algorithm:

1. **The Actor (Policy Network $\pi_\theta$):** Takes the 110-float observation vector $o_i$ and outputs a probability distribution (mean $\mu$ and standard deviation $\sigma$) over continuous 2D waypoint actions. Early in training, $\sigma$ is large, so the drones explore randomly.
2. **The Critic (Value Network $V_\phi$):** Takes the same observation $o_i$ and predicts the **expected cumulative future reward** $G_t = \sum_{k=0}^{T} \gamma^k r_{t+k}$ (where $\gamma = 0.99$ is the discount factor that makes the drone care about long-term map completion, not just the next step).
3. **Advantage Calculation ($A_t$):** After collecting a batch of experience (e.g., 1,024 steps), PPO compares what actually happened ($G_t$) against what the Critic expected ($V_\phi(o_t)$):

$$A_t = G_t - V_\phi(o_t)$$



If $A_t > 0$, the action led to higher coverage than expected, so gradient ascent increases the probability of taking that action in similar states. If $A_t < 0$, it decreases the probability. PPO "clips" the gradient update so a single noisy batch never destroys what the network has already learned.

### Step 3: Why CTDE (Centralized Training, Decentralized Execution) Works

Training $N$ completely separate neural networks from scratch is slow and unstable because every drone's environment keeps changing as its teammates change behavior (non-stationarity). Instead, we use **Parameter Sharing**:

* **During Training:** All 4 drones feed their local observations $o_i$ into **one shared neural network** $\pi_\theta$, pooling 4× as much experience per simulation step.
* **During Execution:** Each drone runs its own independent copy of the frozen weights $\theta$ onboard. Because neighbor inputs are sorted by distance and padded when a drone drops out, the network naturally learns: *"When a neighbor slot disappears from my input vector, an unvisited region will open up in that direction—fly toward the high-uncertainty cells there."*

---

As you transition into research, would you like to learn how to set up a **quantitative benchmarking pipeline** (comparing coverage speed, path overlap, and failure-recovery time across algorithms with statistical plots), or explore **Graph Neural Networks (GNNs) / Attention** so the RL policy can handle arbitrary numbers of neighbors without a fixed 3-neighbor limit?
