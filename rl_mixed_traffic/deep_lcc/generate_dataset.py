"""Generate (state, solution) dataset for NNMPC training.

Uses the same simulation loop (run_with_state) as eval_classical.py
to ensure training data matches the evaluation environment.

Usage:
    uv run rl_mixed_traffic/deep_lcc/generate_dataset.py
"""

from pathlib import Path

import numpy as np
from tqdm import trange

from rl_mixed_traffic.deep_lcc.config import (
    DeepLCCConfig,
    OVMConfig,
    get_heterogeneous_ovm_config,
)
from rl_mixed_traffic.deep_lcc.eval_classical import run_with_state


def _assign_episode_perturbations(
    num_episodes: int,
    perturb_mix: list[tuple[str, float, float]],
) -> list[tuple[str, float]]:
    """Assign a perturbation (type, amplitude) to each episode."""
    assignments: list[tuple[str, float]] = []
    remaining = num_episodes
    for i, (ptype, amp, frac) in enumerate(perturb_mix):
        if i == len(perturb_mix) - 1:
            count = remaining
        else:
            count = round(num_episodes * frac)
            remaining -= count
        assignments.extend([(ptype, amp)] * count)
    return assignments


def _make_head_velocity(
    ptype: str,
    amplitude: float,
    total_steps: int,
    tstep: float,
    v_star: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate absolute head vehicle velocity for one episode."""
    if ptype == "random":
        return v_star + (-amplitude + 2.0 * amplitude * rng.random(total_steps))

    elif ptype == "brake":
        head_vel = np.full(total_steps, v_star)
        v = v_star
        for k in range(total_steps):
            t = k * tstep
            if t < 2.0:
                a = -5.0
            elif t < 7.0:
                a = 0.0
            elif t < 12.0:
                a = 2.0
            else:
                a = 0.0
            v = max(0.0, v + a * tstep)
            head_vel[k] = v
        return head_vel

    elif ptype == "sinusoidal":
        t = np.arange(total_steps) * tstep
        return v_star + amplitude * np.sin(2.0 * np.pi / 10.0 * t)

    else:
        raise ValueError(f"Unknown perturbation type: {ptype}")


def _build_weight_matrices(
    config: DeepLCCConfig, n_vehicle: int, m_ctr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Q and R weight matrices for the DeeP-LCC cost."""
    Q_v = config.weight_v * np.eye(n_vehicle)
    if config.measure_type == 3:
        Q_s = config.weight_s * np.eye(m_ctr)
        Q = np.block(
            [
                [Q_v, np.zeros((n_vehicle, m_ctr))],
                [np.zeros((m_ctr, n_vehicle)), Q_s],
            ]
        )
    elif config.measure_type == 2:
        Q_s = config.weight_s * np.eye(n_vehicle)
        Q = np.block(
            [
                [Q_v, np.zeros((n_vehicle, n_vehicle))],
                [np.zeros((n_vehicle, n_vehicle)), Q_s],
            ]
        )
    else:
        Q = Q_v

    R = config.weight_u * np.eye(m_ctr)
    return Q, R


def generate_dataset(
    config: DeepLCCConfig | None = None,
    ovm_config: OVMConfig | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate dataset using run_with_state (same simulation as eval)."""
    if config is None:
        config = DeepLCCConfig()
    if ovm_config is None:
        ovm_config = get_heterogeneous_ovm_config()

    n_vehicle = ovm_config.n_vehicle
    pos_cav = np.where(np.array(ovm_config.ID) == 1)[0]
    m_ctr = len(pos_cav)

    Q, R = _build_weight_matrices(config, n_vehicle, m_ctr)
    episode_perturbations = _assign_episode_perturbations(
        config.num_episodes,
        config.perturb_mix,
    )
    total_steps = int(config.total_time / config.Tstep)

    all_uini = []
    all_yini = []
    all_eini = []
    all_u_opt = []

    for episode in trange(config.num_episodes, desc="Generating dataset"):
        ep_seed = seed + episode
        ptype, amp = episode_perturbations[episode]

        rng = np.random.default_rng(ep_seed + 10000)
        head_vel = _make_head_velocity(
            ptype, amp, total_steps, config.Tstep, config.v_star, rng,
        )

        _, _, _, dataset_pairs = run_with_state(
            config, ovm_config, Q, R, head_vel,
            seed=ep_seed,
            noise_seed=ep_seed + 20000,
            enable_aeb=False,
            update_s_star=False,
            collect_dataset=True,
        )

        if dataset_pairs is not None:
            all_uini.extend(dataset_pairs["uini"])
            all_yini.extend(dataset_pairs["yini"])
            all_eini.extend(dataset_pairs["eini"])
            all_u_opt.extend(dataset_pairs["u_opt"])

    dataset = {
        "uini": np.array(all_uini),
        "yini": np.array(all_yini),
        "eini": np.array(all_eini),
        "u_opt": np.array(all_u_opt),
        "metadata": np.array(
            [
                config.v_star,
                config.s_star,
                config.T_ini,
                config.N,
                config.lambda_g,
                config.lambda_y,
            ]
        ),
    }

    return dataset


def main():
    config = DeepLCCConfig()
    ovm_config = get_heterogeneous_ovm_config()

    dataset = generate_dataset(config, ovm_config)

    # Save
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **dataset)

    n_samples = dataset["uini"].shape[0]
    print(f"Dataset saved to {out_path}")
    print(f"  Samples: {n_samples}")
    print(f"  uini shape: {dataset['uini'].shape}")
    print(f"  yini shape: {dataset['yini'].shape}")
    print(f"  eini shape: {dataset['eini'].shape}")
    print(f"  u_opt shape: {dataset['u_opt'].shape}")


if __name__ == "__main__":
    main()
