# Project Refactoring Plan

**Project**: SUMO RL Mixed Traffic Control
**Date Created**: 2025-11-28
**Last Updated**: 2025-11-28
**Status**: 🚧 In Progress - Phase 1 Complete (All Agents Refactored)
**Scope**: Comprehensive code refactoring to improve architecture, maintainability, and extensibility

---

## 📊 Progress Summary

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1.1: Fix BaseAgent | ✅ Complete | 100% |
| Phase 1.2: Refactor QLearningAgent | ✅ Complete | 100% |
| Phase 1.3: Refactor DQNAgent | ✅ Complete | 100% |
| Phase 2: Environment Refactoring | ⏳ Pending | 0% |
| Phase 3: Training Infrastructure | ⏳ Pending | 0% |
| Phase 4: Configuration & Utilities | ⏳ Pending | 0% |
| Phase 5: Testing & Documentation | ⏳ Pending | 0% |
| Phase 6: Advanced Features | ⏳ Pending | 0% |

**Overall Progress**: 3/8 phases complete (37.5%)

---

## 📝 Changelog

### 2025-11-28 - Phase 1 Complete (All Phases: 1.1, 1.2, 1.3)

**Phase 1.1: BaseAgent Implementation**
- Fixed import typo in `base_agent.py` (`abs` → `abc`)
- Established complete abstract interface for all agents
- Defined standard API: `act()`, `update()`, `save()`, `load()`, `get_config()`

**Phase 1.2: QLearningAgent Refactoring**
- Created `rl_mixed_traffic/configs/q_config.py` with QLearningConfig dataclass
- Fixed critical Q-learning bug in TD target calculation
- Implemented all abstract methods (save, load, get_config)
- Enhanced `act()` to accept both tuple and np.ndarray states
- Added comprehensive documentation and type hints
- Verified backward compatibility with existing training scripts

**Phase 1.3: DQNAgent Refactoring**
- Added `update()` method (unified interface wrapping store_transition + learn)
- Implemented `get_config()` method returning complete agent configuration
- Enhanced `save()` to include optimizer state and config metadata
- Enhanced `load()` to restore optimizer state for training resumption
- Added comprehensive docstrings explaining Double DQN algorithm
- Added complete type hints throughout
- Maintained 100% backward compatibility (old API still works)

**Files Modified**:
- `rl_mixed_traffic/agents/base_agent.py` (1 line - typo fix)
- `rl_mixed_traffic/agents/q_agent.py` (complete refactor, ~230 lines)
- `rl_mixed_traffic/agents/dqn_agent.py` (complete refactor, ~280 lines)
- `rl_mixed_traffic/configs/q_config.py` (new file, ~30 lines)

**Backward Compatibility**: ✅ 100% - All existing code continues to work

---

## Executive Summary

This document outlines a systematic refactoring plan for the SUMO RL Mixed Traffic project. The refactoring aims to:
- Establish proper inheritance hierarchies (BaseAgent is incomplete)
- Improve code organization and separation of concerns
- Enhance configurability and reduce code duplication
- Maintain backward compatibility with existing trained models
- Improve testability and documentation

---

## Current Architecture Issues

### 1. **Incomplete Agent Abstraction** ~~(FULLY RESOLVED)~~ ✅
- ~~[base_agent.py](rl_mixed_traffic/agents/base_agent.py:1) has a typo~~ ✅ **FIXED**
- ~~BaseAgent class has no abstract methods defined~~ ✅ **FIXED**
- ~~QLearningAgent doesn't inherit from BaseAgent~~ ✅ **FIXED**
- ~~DQNAgent doesn't inherit from BaseAgent~~ ✅ **FIXED**
- ~~No unified interface for agent operations~~ ✅ **FIXED - All agents implement BaseAgent**

### 2. **Environment Concerns**
- [ring_env.py](rl_mixed_traffic/env/ring_env.py) has mixed responsibilities:
  - SUMO simulation management
  - Reward computation (hardcoded in rl_mixed_traffic/env/ring_env.py:277)
  - Head vehicle behavior control
  - State observation construction
- Reward function has magic numbers and is not easily configurable
- `time.sleep(0.01)` in step() (rl_mixed_traffic/env/ring_env.py:266) is a code smell
- Comments like `# TODO: Linear interpolation?` (rl_mixed_traffic/env/ring_env.py:58) suggest incomplete features

### 3. **Code Duplication**
- Training loops in [q_train.py](rl_mixed_traffic/q_train.py) and [dqn_train.py](rl_mixed_traffic/dqn_train.py) have similar structure
- Environment creation duplicated across files
- Evaluation scripts likely duplicate setup logic

### 4. **Configuration Management** ~~(PARTIALLY RESOLVED)~~
- Config files are separated but not consistently used
- Some hyperparameters hardcoded in training scripts (e.g., num_bins, num_vehicles)
- ~~No validation for config values~~ ✅ **FIXED for QLearningConfig** (has validation in `__post_init__`)
- ~~Missing config for Q-learning~~ ✅ **FIXED** (created QLearningConfig dataclass)
- ⚠️ Still need unified experiment config and YAML/JSON loading support

### 5. **Testing & Validation**
- No unit tests visible in tests/ directory
- No integration tests for RL agents
- No validation utilities for trained models

### 6. **Type Safety & Documentation**
- Inconsistent type hints across codebase
- Missing docstrings in key methods
- No comprehensive API documentation

---

## Refactoring Phases

### Phase 1: Foundation & Agent Hierarchy (Priority: HIGH)

**Goal**: Establish a proper agent abstraction layer

#### 1.1 Fix BaseAgent Implementation ✅ **COMPLETE**
**Files**: [rl_mixed_traffic/agents/base_agent.py](rl_mixed_traffic/agents/base_agent.py)

**Completed Changes**:
- ✅ Fixed import typo: `from abs import ABC` → `from abc import ABC`
- ✅ Defined complete abstract interface with all required methods
- ✅ Added comprehensive docstrings
- ✅ Established standard agent API (act, update, save, load, get_config)

**Implementation Status**:
```python
from abc import ABC, abstractmethod  # ✅ Fixed
import numpy as np
from typing import Any, Dict, Tuple

class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    @abstractmethod
    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select an action given current state."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Update agent parameters (Q-table, neural network, etc.)."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to disk."""
        pass

    @abstractmethod
    def load(self, path: str, map_location=None) -> None:
        """Load agent from disk."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        pass
```

#### 1.2 Refactor QLearningAgent ✅ **COMPLETE**

**Files**:
- [rl_mixed_traffic/agents/q_agent.py](rl_mixed_traffic/agents/q_agent.py)
- [rl_mixed_traffic/configs/q_config.py](rl_mixed_traffic/configs/q_config.py) (NEW)

**Completed Changes**:
- ✅ QLearningAgent already inherits from BaseAgent
- ✅ Created QLearningConfig dataclass with validation
- ✅ Implemented all abstract methods (save, load, get_config)
- ✅ Fixed critical Q-learning TD target calculation bug
- ✅ Updated act() signature to match BaseAgent (accepts np.ndarray + eval_mode)
- ✅ Added comprehensive type hints and docstrings
- ✅ Maintained 100% backward compatibility with existing code

**Key Improvements**:

1. **QLearningConfig Dataclass** ([q_config.py](rl_mixed_traffic/configs/q_config.py)):
   ```python
   @dataclass
   class QLearningConfig:
       alpha: float = 0.2          # Learning rate
       gamma: float = 0.98         # Discount factor
       eps_start: float = 1.0      # Initial exploration
       eps_end: float = 0.1        # Final exploration
       eps_decay_steps: int = 15_000
       action_space: int = 7

       def __post_init__(self):
           # Validates all parameters are in valid ranges
   ```

2. **Bug Fix - TD Target Calculation**:
   ```python
   # OLD (INCORRECT) - line 51-53:
   max_next_q = 0 if done else reward + self.gamma * np.max(self.q_table[next_state])
   # This incorrectly set max_next_q = 0 when done, ignoring reward

   # NEW (CORRECT) - lines 140-143:
   if done:
       td_target = reward
   else:
       td_target = reward + self.gamma * np.max(self.q_table[next_state])
   # Now correctly uses reward in terminal states
   ```

3. **Enhanced save/load Methods**:
   - `save()`: Saves Q-table + metadata (step_count, config)
   - `load()`: Loads with validation for action space compatibility
   - Proper serialization of defaultdict to plain dict

4. **Flexible State Handling**:
   - `act()` now accepts both tuple and np.ndarray states
   - `update()` converts states automatically for consistency
   - Maintains backward compatibility with tuple-based interface

**Testing Results**:
- ✅ All imports successful
- ✅ Backward compatible with existing q_train.py
- ✅ Save/load functionality verified
- ✅ Both old (individual params) and new (config) interfaces work
- ✅ Q-table checkpoint format enhanced but still loads old files

#### 1.3 Refactor DQNAgent ✅ **COMPLETE**

**Files**: [rl_mixed_traffic/agents/dqn_agent.py](rl_mixed_traffic/agents/dqn_agent.py)

**Completed Changes**:
- ✅ DQNAgent already inherits from BaseAgent
- ✅ Added `update()` method (wraps store_transition + learn)
- ✅ Kept `store_transition()` and `learn()` as separate methods (backward compatibility)
- ✅ Implemented `get_config()` method
- ✅ Enhanced `save()` to include optimizer state and config
- ✅ Enhanced `load()` to restore optimizer state
- ✅ Added comprehensive documentation and type hints
- ✅ Maintained 100% backward compatibility

**Key Improvements**:

1. **Unified update() Method**:
   ```python
   def update(self, state, action, reward, next_state, done) -> Optional[float]:
       """Store transition and perform learning update (if conditions are met).

       This is a unified interface that:
       1. Stores the transition in replay buffer
       2. Performs a learning step if conditions are met
       """
       self.store_transition(state, action, reward, next_state, done)
       return self.learn()
   ```

2. **Complete get_config() Implementation**:
   - Returns all hyperparameters (lr, gamma, batch_size, etc.)
   - Includes training state (steps, buffer_size_current)
   - Useful for experiment tracking and reproducibility

3. **Enhanced Checkpointing**:
   ```python
   # save() now includes:
   - Q-network state dict
   - Target network state dict
   - Optimizer state dict (NEW)
   - Training steps
   - Complete config (NEW)
   ```

4. **Backward Compatibility**:
   - Old API still works: `agent.buffer.append()` + `agent.learn()`
   - New API available: `agent.update(state, action, reward, next_state, done)`
   - No changes required to existing dqn_train.py

**Testing Results**:
- ✅ Import successful
- ✅ Agent creation works
- ✅ get_config() returns complete configuration
- ✅ act() with eval_mode parameter works
- ✅ update() (new unified API) works
- ✅ store_transition() + learn() (old API) still works
- ✅ save() includes optimizer and config
- ✅ load() restores complete state
- ✅ 100% backward compatible with dqn_train.py

---

### Phase 2: Environment Refactoring (Priority: HIGH)

**Goal**: Separate concerns and make environment more modular

#### 2.1 Extract Reward Function
**Files**:
- New: `rl_mixed_traffic/env/rewards.py`
- Modify: [rl_mixed_traffic/env/ring_env.py](rl_mixed_traffic/env/ring_env.py:277)

**Tasks**:
- [ ] Create RewardConfig dataclass with all reward parameters:
  ```python
  @dataclass
  class RewardConfig:
      # TTC penalty
      ttc_threshold: float = 0.6
      ttc_weight: float = 0.15
      ttc_penalty_base: float = -0.02

      # Headway distance
      gap_threshold: float = 15.0
      gap_penalty: float = -1.0
      gap_weight: float = 1.0

      # Jerk penalty
      jerk_weight: float = 0.2

      # Safety margins
      d_min: float = 2.0
  ```
- [ ] Create abstract `BaseReward` class
- [ ] Implement `MultiComponentReward` class using composition
- [ ] Allow reward function to be injected into RingRoadEnv
- [ ] Remove hardcoded reward from compute_reward() in ring_env.py:277

#### 2.2 Extract Head Vehicle Controller
**Files**:
- New: `rl_mixed_traffic/env/head_vehicle_controller.py`
- Modify: [rl_mixed_traffic/env/ring_env.py](rl_mixed_traffic/env/ring_env.py:105)

**Tasks**:
- [ ] Create HeadVehicleController class
- [ ] Support multiple strategies (random speed changes, sinusoidal, constant, etc.)
- [ ] Make controller configurable and injectable
- [ ] Remove _update_head_speed() from RingRoadEnv

#### 2.3 Environment Configuration
**Files**:
- New: `rl_mixed_traffic/configs/env_config.py`
- Modify: [rl_mixed_traffic/env/ring_env.py](rl_mixed_traffic/env/ring_env.py)

**Tasks**:
- [ ] Create EnvConfig dataclass consolidating all env parameters
- [ ] Replace individual __init__ parameters with config object
- [ ] Add validation for config values
- [ ] Remove magic numbers (v_max=30.0, etc.)

#### 2.4 Clean Up Environment Code
**Files**: [rl_mixed_traffic/env/ring_env.py](rl_mixed_traffic/env/ring_env.py)

**Tasks**:
- [ ] Remove `time.sleep(0.01)` from step() (line 266) - replace with proper synchronization
- [ ] Address TODO comment at line 58
- [ ] Extract state normalization into separate method
- [ ] Remove debug print statements (line 120, 328)
- [ ] Improve error handling in compute_reward() leader detection (lines 286-313)
- [ ] Make TraCI connection management more robust

---

### Phase 3: Training Infrastructure (Priority: MEDIUM)

**Goal**: Unify training loops and reduce duplication

#### 3.1 Create Unified Trainer
**Files**:
- New: `rl_mixed_traffic/training/trainer.py`
- New: `rl_mixed_traffic/training/callbacks.py`

**Tasks**:
- [ ] Create abstract `BaseTrainer` class
- [ ] Implement `EpisodicTrainer` (for Q-learning)
- [ ] Implement `StepBasedTrainer` (for DQN)
- [ ] Support callbacks for logging, checkpointing, early stopping
- [ ] Create `TrainingConfig` dataclass

#### 3.2 Refactor Training Scripts
**Files**:
- [rl_mixed_traffic/q_train.py](rl_mixed_traffic/q_train.py)
- [rl_mixed_traffic/dqn_train.py](rl_mixed_traffic/dqn_train.py)

**Tasks**:
- [ ] Migrate to unified Trainer API
- [ ] Use consistent config management
- [ ] Remove duplicated plotting logic
- [ ] Add proper argument parsing (click or argparse)
- [ ] Support resuming training from checkpoints

#### 3.3 Evaluation Framework
**Files**:
- New: `rl_mixed_traffic/evaluation/evaluator.py`
- Modify: `rl_mixed_traffic/q_eval_policy.py`
- Modify: `rl_mixed_traffic/dqn_eval.py`

**Tasks**:
- [ ] Create unified Evaluator class
- [ ] Support multiple evaluation metrics (returns, collision rate, avg speed, etc.)
- [ ] Add visualization utilities (trajectory plots, speed diagrams)
- [ ] Create evaluation reports (JSON/CSV output)

---

### Phase 4: Configuration & Utilities (Priority: MEDIUM)

#### 4.1 Unified Config System
**Files**:
- New: `rl_mixed_traffic/configs/base_config.py`
- New: `rl_mixed_traffic/configs/experiment_config.py`

**Tasks**:
- [ ] Create `BaseConfig` with validation logic
- [ ] Create `ExperimentConfig` that combines all sub-configs
- [ ] Support config loading from YAML/JSON files
- [ ] Add config validation and error reporting
- [ ] Create config inheritance/composition system

#### 4.2 Logging & Monitoring
**Files**:
- New: `rl_mixed_traffic/utils/logger.py`
- New: `rl_mixed_traffic/utils/metrics.py`

**Tasks**:
- [ ] Create structured logging system (replace print statements)
- [ ] Support TensorBoard/WandB integration
- [ ] Create metrics tracker for training/evaluation
- [ ] Add checkpointing utilities

#### 4.3 Discretization Improvements
**Files**: [rl_mixed_traffic/env/discretizer.py](rl_mixed_traffic/env/discretizer.py)

**Tasks**:
- [ ] Make StateDiscretizer more flexible (arbitrary obs shapes)
- [ ] Support different discretization strategies (uniform, exponential, custom)
- [ ] Add discretization visualization tools
- [ ] Consider making DiscretizeActionWrapper more generic

---

### Phase 5: Testing & Documentation (Priority: MEDIUM)

#### 5.1 Unit Tests
**Files**: New files in `tests/`

**Tasks**:
- [ ] Test agent implementations (act, update, save/load)
- [ ] Test environment (reset, step, reward computation)
- [ ] Test discretizers (action/state)
- [ ] Test config validation
- [ ] Test SUMO utilities

#### 5.2 Integration Tests
**Files**: New files in `tests/integration/`

**Tasks**:
- [ ] Test full training loops (short runs)
- [ ] Test checkpoint save/load/resume
- [ ] Test environment-agent interaction
- [ ] Test SUMO connection handling

#### 5.3 Documentation
**Files**: Various

**Tasks**:
- [ ] Add comprehensive docstrings (Google style)
- [ ] Create API documentation (Sphinx)
- [ ] Update CLAUDE.md with new architecture
- [ ] Create architecture diagrams
- [ ] Write migration guide for existing code

---

### Phase 6: Advanced Features (Priority: LOW)

#### 6.1 Multi-Agent Support
**Files**: TBD

**Tasks**:
- [ ] Extend environment for multiple RL agents
- [ ] Implement multi-agent coordination
- [ ] Support heterogeneous agent types

#### 6.2 Additional Algorithms
**Files**: `rl_mixed_traffic/agents/`

**Tasks**:
- [ ] Implement PPO (Actor-Critic mentioned in ring_env.py:346)
- [ ] Implement SAC
- [ ] Implement A2C/A3C
- [ ] Create algorithm comparison framework

#### 6.3 Advanced Scenarios
**Files**: `configs/`, `rl_mixed_traffic/env/`

**Tasks**:
- [ ] Support multiple road geometries (not just ring)
- [ ] Variable traffic density
- [ ] Heterogeneous vehicle types
- [ ] Real-world map integration

---

## Migration Strategy

### Backward Compatibility

To ensure existing experiments can continue:

1. **Preserve Old Interfaces (Temporarily)**
   - Keep old training scripts as `q_train_legacy.py`, `dqn_train_legacy.py`
   - Add deprecation warnings
   - Provide 2-version grace period

2. **Model Compatibility**
   - Ensure new agent classes can load old checkpoints
   - Add checkpoint version metadata
   - Provide conversion scripts if needed

3. **Config Migration**
   - Provide script to convert old hardcoded configs to new YAML format
   - Auto-migration on load with warnings

### Rollout Plan

1. **Week 1-2**: Phase 1 (Agent Hierarchy)
2. **Week 3-4**: Phase 2 (Environment Refactoring)
3. **Week 5-6**: Phase 3 (Training Infrastructure)
4. **Week 7**: Phase 4 (Config & Utilities)
5. **Week 8-9**: Phase 5 (Testing & Documentation)
6. **Week 10+**: Phase 6 (Advanced Features - ongoing)

---

## Code Quality Standards

### Python Style
- Follow PEP 8
- Use `ruff` for linting (already in dev dependencies)
- Maximum line length: 100 characters
- Use type hints everywhere
- Google-style docstrings

### Testing Requirements
- Minimum 80% code coverage
- All public APIs must have tests
- Integration tests for critical paths

### Documentation Requirements
- All public classes/functions must have docstrings
- Module-level docstrings explaining purpose
- Examples in docstrings for complex APIs

---

## Risk Assessment

### High Risk
- **SUMO Integration Breakage**: Refactoring environment could break SUMO connection
  - *Mitigation*: Extensive integration testing, feature flags

- **Model Compatibility**: Breaking changes to agent save/load
  - *Mitigation*: Checkpoint versioning, conversion utilities

### Medium Risk
- **Performance Regression**: Abstraction overhead
  - *Mitigation*: Benchmarking before/after, profiling

- **API Confusion**: New interfaces may confuse existing users
  - *Mitigation*: Migration guide, examples, deprecation warnings

### Low Risk
- **Config Complexity**: Over-engineering configuration system
  - *Mitigation*: Start simple, iterate based on needs

---

## Success Metrics

- [ ] All agents inherit from BaseAgent with complete implementation
- [ ] Zero `# TODO` comments in core code
- [ ] Zero hardcoded magic numbers in environment/reward
- [ ] >80% test coverage
- [ ] All public APIs documented
- [ ] Existing trained models still loadable
- [ ] Training time <5% slower than before
- [ ] Can add new RL algorithm in <200 lines of code

---

## Open Questions

1. Should we support other traffic simulators (VISSIM, Aimsun)?
2. Do we need distributed training support?
3. Should reward functions be learnable (inverse RL)?
4. Integration with existing RL libraries (Stable Baselines3, RLlib)?
5. Should we support curriculum learning?

---

## Appendix: File Structure After Refactoring

```
rl_mixed_traffic/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # ✅ Fixed BaseAgent
│   ├── q_agent.py             # ✅ Inherits BaseAgent
│   ├── dqn_agent.py           # ✅ Inherits BaseAgent
│   └── ppo_agent.py           # 🆕 Future
├── configs/
│   ├── __init__.py
│   ├── base_config.py         # 🆕 Config base class
│   ├── env_config.py          # 🆕 Environment config
│   ├── reward_config.py       # 🆕 Reward config
│   ├── agent_config.py        # 🆕 Agent config base
│   ├── q_config.py            # 🆕 Q-learning config
│   ├── dqn_config.py          # ✅ Existing
│   ├── sumo_config.py         # ✅ Existing
│   └── experiment_config.py   # 🆕 Full experiment config
├── env/
│   ├── __init__.py
│   ├── ring_env.py            # ✅ Refactored, cleaner
│   ├── discretizer.py         # ✅ Improved
│   ├── rewards.py             # 🆕 Reward functions
│   └── head_vehicle_controller.py  # 🆕 Head vehicle logic
├── training/
│   ├── __init__.py
│   ├── trainer.py             # 🆕 Unified trainer
│   ├── callbacks.py           # 🆕 Training callbacks
│   └── checkpointing.py       # 🆕 Checkpoint utilities
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           # 🆕 Unified evaluator
│   ├── metrics.py             # 🆕 Evaluation metrics
│   └── visualizer.py          # 🆕 Visualization tools
├── utils/
│   ├── __init__.py
│   ├── logger.py              # 🆕 Structured logging
│   ├── metrics.py             # 🆕 Metrics tracking
│   ├── plot_utils.py          # ✅ Existing
│   ├── reward.py              # ✅ Existing (maybe merge with env/rewards.py)
│   └── sumo_utils.py          # ✅ Existing
├── dqn/
│   ├── __init__.py
│   ├── network.py             # ✅ Existing
│   └── replay_mem.py          # ✅ Existing
├── scripts/
│   ├── __init__.py
│   ├── train.py               # 🆕 New unified training script
│   ├── evaluate.py            # 🆕 New unified eval script
│   └── classic_controller.py  # ✅ Existing
├── q_train.py                 # ⚠️ Deprecated (keep for compatibility)
├── q_eval_policy.py           # ⚠️ Deprecated
├── dqn_train.py               # ⚠️ Deprecated
└── dqn_eval.py                # ⚠️ Deprecated
```

---

## References

- SUMO Documentation: https://sumo.dlr.de/docs/
- OpenAI Gym/Gymnasium: https://gymnasium.farama.org/
- Deep RL Best Practices: https://stable-baselines3.readthedocs.io/
- CLAUDE.md: Current project documentation
