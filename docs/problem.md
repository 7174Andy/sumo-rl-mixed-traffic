# Problem Formulation

## 1.1 Environment: Ring Road Mixed Traffic

This project studies a **single-lane ring road** where one RL-controlled autonomous vehicle (CAV) shares the road with a human-driven head vehicle. The ring topology eliminates boundary effects (on-ramps, traffic lights) and isolates the core car-following dynamics.

The head vehicle (`car0`) periodically changes its cruising speed at random, creating disturbances that propagate through the ring. The AV (`car1`) must learn a speed-control policy that maintains safe following distances, tracks the traffic flow, and avoids abrupt acceleration changes.

The problem is formulated as a **Markov Decision Process (MDP)**:

| Component      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| **State**      | Normalized velocities and positions of all vehicles on the ring             |
| **Action**     | Longitudinal acceleration command applied to the CAV                        |
| **Reward**     | Quadratic cost penalizing velocity error, spacing error, and control effort |
| **Transition** | Determined by SUMO's car-following models and the applied acceleration      |

The simulation runs at a time step of $\Delta t = 0.1\,\text{s}$, and each episode lasts up to 500 s (5,000 steps).

---

## 1.2 State Space

The observation is a fixed-length vector of normalized velocities and positions for all $N$ vehicles:

$$
\mathbf{s} = \bigl[\underbrace{v_0/v_{\max},\;\dots,\;v_{N-1}/v_{\max}}_{\text{velocities}},\;\underbrace{p_0/L,\;\dots,\;p_{N-1}/L}_{\text{positions}}\bigr]
$$

where:

- $v_{\max} = 30\;\text{m/s}$ is the maximum speed.
- $L$ is the ring circumference (computed from the SUMO network at startup).
- Vehicles are **sorted by ID** so each index always maps to the same vehicle.
- If fewer than $N$ vehicles are present, the vector is **zero-padded** to maintain a fixed shape of $(2N,)$.
- All values are clipped to $[0, 1]$.

For the default two-vehicle scenario, the observation has shape $(4,)$: `[v_head, v_agent, p_head, p_agent]`.

---

## 1.3 Action Space

The native action space is a **continuous scalar acceleration**:

$$
a \in [a_{\min},\; a_{\max}] = [-3.0,\; 3.0]\;\text{m/s}^2
$$

The acceleration is integrated into velocity each step:

$$
v_{t+1} = \text{clip}\bigl(v_t + a \cdot \Delta t,\; 0,\; v_{\max}\bigr)
$$

and applied to the vehicle via TraCI's `setSpeed()` for immediate response.

### Discretization

Because both Q-learning and DQN operate over discrete actions, the continuous range is discretized into $n$ evenly spaced bins using `DiscretizeActionWrapper`:

$$
\mathcal{A} = \bigl\{a_{\min},\;\; a_{\min} + \delta,\;\; a_{\min} + 2\delta,\;\;\dots,\;\; a_{\max}\bigr\}, \quad \delta = \frac{a_{\max} - a_{\min}}{n - 1}
$$

---

## 1.4 Reward Function

The active reward function is based on the **DeeP-LCC** [[1]](#references) cost formulation, which penalizes deviations from equilibrium velocity, desired spacing, and excessive control effort.

### Components

**Velocity error** — penalizes deviation from the target speed ($v^{*} = 15\;\text{m/s}$):

$$
R_v = -w_v \cdot (v_{\text{ego}} - v^{*})^2
$$

**Spacing error** — penalizes deviation from the OVM equilibrium spacing $s^{*}$:

$$
R_s = -w_s \cdot \bigl(\text{clip}(d_{\text{gap}} - s^{*},\;-20,\;20)\bigr)^2
$$

where $s^{*}$ is computed using the Optimal Velocity Model (OVM):

$$
s^{*} = \frac{\arccos\!\bigl(1 - 2\,v^{*}/v_{\max}\bigr)}{\pi}\,(s_{\text{go}} - s_{\text{st}}) + s_{\text{st}}
$$

with $s_{\text{st}} = 5\;\text{m}$ (stop spacing) and $s_{\text{go}} = 35\;\text{m}$ (free-flow spacing).

**Control effort** — penalizes large accelerations for smooth driving:

$$
R_u = -w_u \cdot a^2
$$

**Safety constraint** — a hard penalty when the gap falls below a minimum:

$$
R_{\text{safety}} = \begin{cases} -100 & \text{if } d_{\text{gap}} < s_{\min} \\ 0 & \text{otherwise} \end{cases}
$$

### Total Reward

$$
R = \frac{R_v + R_s + R_u + R_{\text{safety}}}{100}
$$

The division by 100 scales per-step values into a range suitable for policy gradient methods.

### Default Weights

| Parameter | Symbol     | Default |
| --------- | ---------- | ------- |
| Velocity  | $w_v$      | 0.8     |
| Spacing   | $w_s$      | 0.7     |
| Control   | $w_u$      | 0.1     |
| Min gap   | $s_{\min}$ | 5.0 m   |

## References

[1] Wang, J., Zheng, Y., Li, K., & Xu, Q. (2023). DeeP-LCC: Data-EnablEd Predictive Leading Cruise Control in Mixed Traffic Flow. IEEE Transactions on Control Systems Technology, 31(6), 2760–2776. doi:10.1109/tcst.2023.3288636
