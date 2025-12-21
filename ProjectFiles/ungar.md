# Universal Neural Grid for Analysis and Research (UNGAR)

Implementation of [VISION.md](VISION.md).

## Project Structure

```text
src/ungar/
├── cards.py      # Core Card/Suit/Rank primitives
├── enums.py      # Constants (SUIT_COUNT=4, RANK_COUNT=14)
├── game.py       # GameState, GameSpec, Move protocols
├── runner.py     # Simulation runner (play_random_episode)
├── tensor.py     # 4x14xN CardTensor implementation
├── xai.py        # Card overlay and XAI data structures
└── games/        # Game implementations
    └── high_card_duel.py

bridge/
├── src/ungar_bridge/
│   ├── adapter_base.py    # BridgeAdapter interface
│   ├── rediai_adapter.py  # RediAI EnvAdapter integration
│   ├── rediai_training.py # Training workflow integration
│   ├── rl_adapter.py      # Gym-like environment wrapper
│   └── xai_overlays.py    # Overlay generators
└── examples/
    ├── demo_rediai.py     # RediAI workflow demo
    └── train_high_card_rediai.py # Training CLI
```

## Data Schema

### Core Tensor
*   **Dimensions:** 4 × 14 × N
    *   **Axis 0 (Suits):** SPADES, HEARTS, DIAMONDS, CLUBS
    *   **Axis 1 (Ranks):** ACE, TWO...KING, JOKER
    *   **Axis 2 (Planes):** Game-specific feature planes (N)
*   **Dtype:** Boolean (True = card present in plane).

### Game Interfaces
*   **GameState:** Protocol defining `current_player`, `legal_moves`, `apply_move`, `is_terminal`, `returns`, `to_tensor`.
*   **Move:** Simple `(id, name)` tuple.

## Implemented Games

### High Card Duel
*   **Players:** 2
*   **Deck:** Standard 52 (Jokers supported but not used in default deal).
*   **Rules:** Each player gets 1 card. Reveal. High card wins.
*   **Tensor Planes:** `my_hand`, `opponent_hand`, `unseen`.

### Mini Spades (M10)
*   **Players:** 2
*   **Deck:** Standard 52.
*   **Rules:** 5 cards dealt. Spades is trump. Follow suit if possible. +1 per trick, +5 game winner.
*   **Tensor Planes:** `my_hand`, `legal_moves`, `current_trick`, `history`.

### Gin Rummy (M11)
*   **Players:** 2
*   **Deck:** Standard 52.
*   **Rules:** 10 cards dealt. Draw/Discard cycle. Knock/Gin logic.
*   **Tensor Planes:** `my_hand`, `in_discard`, `top_discard`, `legal_draw_discard`, `legal_discard`, `legal_knock`.

## RediAI-Backed Training (Optional)

You can run training loops that optionally integrate with RediAI's workflow registry.

*   **CLI:** `python bridge/examples/train_high_card_rediai.py`
*   **Function:** `ungar_bridge.rediai_training.train_high_card_duel_rediai`

If RediAI is installed, metrics like average reward are logged to the workflow. If not, the script runs in a local-dummy mode, printing a warning but completing successfully.

## Milestones

*   **M00:** Foundation (Repo, CI).
*   **M01:** Core Tensor (4x14 grid, partitioning).
*   **M02:** Game Interfaces & High Card Duel.
*   **M03:** Security & Supply Chain (Bandit, SBOM, Release Flow).
*   **M04:** Bridge Package & External Integration.
*   **M05:** RediAI Integration (Bridge-Level).
*   **M06:** Bridge Quality & Coverage Hardening.
*   **M07:** Codebase Audit & Refinement.
*   **M08:** RediAI Training Workflow & XAI Scaffold (Done).
*   **M09:** Deep RediAI Integration (XAI/RewardLab) & DX Polish (Done).
*   **M10:** Multi-Game Platform Validation with Mini Spades (Done).
*   **M11:** Gin Rummy & Advanced State Encoding (Done).
*   **M12:** Unified Agents & DQN-Lite (Done).
*   **M13:** PPO Algorithm & Config Layer (Done).
*   **M14:** Backend Device & Logging (Done).
*   **M15:** Analytics & Visualization Tools (Done).
*   **M16:** Unified CLI (Done).
*   **M17:** Analytics Contracts & Frontend v1 Freeze (Done).
*   **M18:** CI Stabilization & Security Hardening (Done).
*   **M19:** XAI Overlay Engine v1: Heuristic & Random (Done).
*   **M20:** Gradient-Based XAI (`policy_grad`) + Overlay Comparison (Done).
*   **M21:** Value-Gradient XAI (`value_grad`, PPO-only) (Done).
*   **M22:** Batch Overlay Engine (PPO XAI Performance) (Done).

### M22 Demo

A comprehensive demo script validates the complete M22 batch overlay engine:

```bash
python scripts/demo_m22_full_cycle.py
```

**Phases:**
1. Training with batched XAI (PPO on High Card Duel)
2. Numerical equivalence validation (batch == sequential < 1e-6)
3. CLI commands verification
4. Heatmap visualization
5. Performance profiling (optional with `--with-profiling`)
6. Summary JSON output

See [docs/demo_m22.md](docs/demo_m22.md) for complete documentation and validation criteria.

## v1 Core Contract

**The interfaces described in [`docs/api_contracts_v1.md`](docs/api_contracts_v1.md) represent the v1 core contract for UNGAR.**

This contract is considered **stable and production-ready** as of M23. Future milestones must treat changes to these interfaces as potentially breaking:

- **4×14×n card tensor representation**
- **CardOverlay structure and XAI methods**
- **Analytics schema v1** (`manifest.json`, `metrics.csv`, overlay files)
- **CLI commands** (`train`, `list-runs`, `show-run`, `summarize-overlays`)

**Contract guarantees:**
- Within v1.x, no breaking changes to these interfaces
- New features will be additive (backward compatible)
- Breaking changes require major version bump (v2.0)

**Architecture decisions:**
- See [`docs/adr/`](docs/adr/) for rationale behind key design choices
- ADR-001: Analytics schema freeze
- ADR-002: XAI overlay engine design
- ADR-003: Opt-in batch processing
- ADR-004: Per-method buffer architecture

## Frontend Status & Integration

The UNGAR frontend is currently **frozen** at v1 (mock-only).

*   **Status Doc:** [docs/frontend_status.md](docs/frontend_status.md)
*   **Integration Contract:** [docs/analytics_schema.md](docs/analytics_schema.md)
*   Future integration is **out-of-scope** until explicitly re-opened. Frontends must consume artifacts via `ungar export-run`.

## Documentation

*   [bridge/README.md](bridge/README.md): Bridge package usage.
*   [docs/bridge_rediai.md](docs/bridge_rediai.md): Detailed RediAI integration docs.
*   [docs/ungar_agents.md](docs/ungar_agents.md): Unified Agent system (DQN/Random).
*   [docs/training_dqn.md](docs/training_dqn.md): Training guide for DQN.
*   [docs/analytics_schema.md](docs/analytics_schema.md): Analytics artifact schema.
