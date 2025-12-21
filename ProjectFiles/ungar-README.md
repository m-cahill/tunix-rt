# UNGAR: Universal Neural Grid for Analysis and Research

> **“All card games, one brain.”**

UNGAR is a unified research environment for imperfect-information card games, designed to build a general-purpose "card intelligence" substrate. Instead of treating Poker, Spades, and Hearts as separate environments, UNGAR models them as different rule sets over a single **4×14×n tensor representation** of card physics.

See [VISION.md](./VISION.md) for the high-level goals, architectural philosophy, and the "moonshot" vision of a General Card Player.

## Architecture: Core vs Bridge

UNGAR is split into two packages:

1.  **`ungar` (Core)**: Framework-agnostic, pure Python/NumPy logic. Contains cards, tensors, rules, and simulation runner.
2.  **`ungar-bridge` (Bridge)**: Adapter layer connecting UNGAR to external frameworks (e.g., RediAI, Gym, UI).

This ensures the core remains lightweight and portable.

## Quickstart (Core)

(Assuming Python 3.10+)

```bash
# Clone the repository
git clone https://github.com/m-cahill/ungar.git
cd ungar

# Install development dependencies
python -m pip install -r requirements-dev.txt

# Run the full CI suite (lint, types, tests) locally
make ci
```

### Run a Game
Run a sample High Card Duel episode in Python:

```bash
python -c "from ungar.games.high_card_duel import make_high_card_duel_spec; from ungar.runner import play_random_episode; print(play_random_episode(make_high_card_duel_spec(), seed=42))"
```

## Core Card Tensor

UNGAR represents the 52-card deck (plus optional jokers) as a 4×14×n NumPy tensor:

* Axis 0: suits (Spades, Hearts, Diamonds, Clubs)
* Axis 1: ranks (Ace through King, plus Joker)
* Axis 2: feature planes (game-defined)

```python
from ungar.cards import Card
from ungar.enums import Suit, Rank
from ungar.tensor import CardTensor

my_hand = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)]
tensor = CardTensor.from_plane_card_map({"my_hand": my_hand})

assert set(tensor.cards_in_plane("my_hand")) == set(my_hand)
```

## Local CI

To run the same checks that GitHub Actions runs:

```bash
python -m pip install -r requirements-dev.txt
make ci
```

## Local dev (pre-commit)

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # run all checks
```

### Run all checks (Linux/macOS/Windows)

```bash
python scripts/run_all_checks.py
```

## RediAI Integration (M09)

M09 adds deep integration with RediAI's XAI and RewardLab ecosystems.

*   **XAI Overlays:** Training runs can now log "explanation" artifacts (card attribution maps) to the RediAI registry.
*   **Reward Decomposition:** Rewards are broken down into components (win/loss, baseline) and logged for RewardLab analysis.
*   See [docs/bridge_rediai.md](docs/bridge_rediai.md) for details.

## Agents & RL (M12)

M12 introduces a unified agent system and the first real RL algorithm: **DQN-Lite**.
*   **UnifiedAgent:** Protocol for swappable agents.
*   **DQN-Lite & PPO-Lite:** Lightweight, PyTorch-based agents that learn on raw card tensors.
*   See [docs/ungar_agents.md](docs/ungar_agents.md), [docs/training_dqn.md](docs/training_dqn.md), and [docs/training_ppo.md](docs/training_ppo.md).
*   See [docs/analytics_overview.md](docs/analytics_overview.md) for M15 analysis tools.

## Mini Spades (M10)

M10 introduces a second game environment: **Mini Spades**. This trick-taking game validates the platform's generality and cross-game XAI capabilities.

## Gin Rummy (M11)

M11 adds **Gin Rummy**, a complex imperfect-information game with draw/discard mechanics, sets/runs meld logic, and knocking. It fully exercises the tensor substrate's ability to represent multi-zone states.

## Roadmap

* **M00** – Repo & CI Bootstrap (Done).
* **M01** – UNGAR core card-physics abstraction + 4×14×n tensor encoder (Done).
* **M02** – Game definitions (High Card Duel) & runner interface (Done).
* **M03** – Security & supply-chain hardening: Bandit, pip-audit, cyclonedx SBOM, basic SLSA provenance for artifacts (Done).
* **M04** – Bridge Package & External Integration (Done).
* **M05** – RediAI Integration (Bridge-Level) (Done).
* **M06** – Bridge Quality & Coverage Hardening (Done).
* **M07** – Codebase Audit & Refinement (Done).
* **M08** – RediAI Training Workflow & XAI Scaffold (Done).
* **M09** – Deep RediAI Integration (XAI/RewardLab) (Done).
* **M10** – Multi-Game Platform Validation with Mini Spades (Done).
* **M11** – Gin Rummy & Advanced State Encoding (Done).
* **M12** – Unified Agents & DQN-Lite (Done).
* **M13** – PPO Algorithm & Config Layer (Done).
* **M14** – Backend Device & Logging (Done).
* **M15** – Analytics & Visualization Tools (Done).
* **M16** – Unified CLI (Done).
* **M17** – Analytics Contracts & Frontend v1 Freeze (Done).
* **M18** – CI Stabilization & Security Hardening (Done).
* **M19** – XAI Overlay Engine v1: Heuristic & Random (Done).
* **M20** – Gradient-Based XAI (`policy_grad`) + Overlay Comparison (Done).
* **M21** – Value-Gradient XAI (`value_grad`, PPO-only) (Done).
* **M22** – Batch Overlay Engine (PPO XAI Performance) (Done).
* **M23** – v1 Core Lock-In & Polish (API Contracts, Versioning, Guardrails) (Done).

## XAI & M22 Batch Overlay Demo

UNGAR includes a complete Explainable AI (XAI) system with batched gradient overlay generation (M22).

**Try the full demo:**

```bash
python scripts/demo_m22_full_cycle.py
```

This validates:
* ✅ Batched XAI overlay generation (5-10× faster)
* ✅ Numerical equivalence (batch == sequential)
* ✅ CLI commands (`train`, `list-runs`, `summarize-overlays`)
* ✅ Heatmap visualization
* ✅ Performance profiling

See **[docs/demo_m22.md](docs/demo_m22.md)** for complete documentation.

## Documentation

**Getting Started:**
* [docs/quickstart_v1.md](docs/quickstart_v1.md): **Start Here!** 30-minute complete onboarding.
* [docs/demo_m22.md](docs/demo_m22.md): **M22 XAI Demo** — Full cycle validation.

**v1 Contracts & Architecture:**
* [docs/api_contracts_v1.md](docs/api_contracts_v1.md): **v1 Core Contract** — Stable interfaces and guarantees.
* [docs/adr/](docs/adr/): Architecture Decision Records (ADR-001 through ADR-004).
* [CHANGELOG.md](CHANGELOG.md): Version history and release notes.

**Core Systems:**
* [docs/cli_reference.md](docs/cli_reference.md): Complete CLI command reference.
* [docs/xai_overlays.md](docs/xai_overlays.md): Explainable AI system guide.
* [docs/analytics_schema.md](docs/analytics_schema.md): Analytics data formats.
* [docs/analytics_overview.md](docs/analytics_overview.md): Analysis and Visualization tools.

**Training & Agents:**
* [docs/ungar_agents.md](docs/ungar_agents.md): Unified Agent system.
* [docs/training_dqn.md](docs/training_dqn.md): DQN training guide.
* [docs/training_ppo.md](docs/training_ppo.md): PPO training guide.

**Integration:**
* [bridge/README.md](bridge/README.md): Bridge package usage.
* [docs/bridge_rediai.md](docs/bridge_rediai.md): RediAI integration.
