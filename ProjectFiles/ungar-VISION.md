Here’s the updated `VISION.md` with the 4×14 spec and the new acronym meaning applied. 

```markdown
Project Vision: Unified Neural Grid for Analysis & Reasoning (UNGAR)
====================================================================

1. High Concept
---------------

**Tagline:**

> **“All card games, one brain.”**

Modern research toolkits like RLCard and OpenSpiel provide multiple card environments behind unified APIs, but still treat each game (Poker, Blackjack, UNO, Gin, Bridge, etc.) as a separate environment with its own bespoke encoding and action logic.([rlcard.org](https://rlcard.org/?utm_source=chatgpt.com "RLCard: A Toolkit for Reinforcement Learning in Card Games ..."))

**UNGAR** (Unified Neural Grid for Analysis & Reasoning) takes a stronger stance:

> **All standard deck card games are instances of a single underlying domain.**  
> Games differ in rules, not in fundamental structure.

The project’s ambition is to build a **general-purpose “card intelligence” substrate** where an AI:

* Learns universal concepts like _suit pressure_, _tempo_, _bluff_, _trick control_, _position_, and _information leakage_.

* Can transfer knowledge across games—e.g., ideas learned from Poker (DeepStack/Pluribus-style bluffing and value betting) to Hearts or Spades (trick management, sandbagging, avoidance of dangerous suits).([arXiv](https://arxiv.org/abs/1701.01724?utm_source=chatgpt.com "DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"))

UNGAR is not “a Poker engine” or “a Spades engine.” It is a **unified research environment for imperfect-information card games**, designed from day one for:

* **Generalization:** one representation across many games.

* **Explainability:** clear, visual insight into _why_ an agent acted.

* **Open science:** fully open-source, reusable by anyone.

* * *

2. Core Philosophy
------------------

### 2.1 Card Physics, Not Game Scripts

Instead of encoding each game as a monolithic pile of rules, UNGAR treats games as **physics plus constraints**:

* **Cards are persistent objects** (rank + suit + identity).

* **Zones are where cards live** (hands, stock, discard, table, tricks, melds, pot, score piles).

* **Actions are movements and declarations**:
  
  * Moving cards between zones (deal, draw, discard, play to trick, reveal).
  
  * Moving resources (chips/points) between players and shared pools.
  
  * Declaring bids, contracts, or special states (nil, knock, bet size choices).

* **Rules constrain those actions in time**:
  
  * Whose turn it is, what’s legal now, how scoring works, when the game ends.

From UNGAR’s perspective, **Texas Hold’em, Spades, Hearts, Gin, Bridge, and custom variants** are all different scripts over the _same_ object model. This echoes how general card game engines and trick-taking frameworks in the literature treat games as rule configurations over a shared core.([courses.cms.caltech.edu](https://courses.cms.caltech.edu/cs145/2013/cards.pdf?utm_source=chatgpt.com "Cards with Friends: A Generic Card Game Engine"))

### 2.2 Imperfect Information as the First-Class Citizen

Poker, Bridge, Hearts, and most interesting card games are **imperfect-information games**. DeepStack and Pluribus showed how powerful specialized algorithms can be in a _single_ such game.([arXiv](https://arxiv.org/abs/1701.01724?utm_source=chatgpt.com "DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"))

UNGAR’s design goal is broader:

> Treat _imperfect information itself_ as the core challenge, not just one specific ruleset.

Everything—from state representation to evaluation and explainability—is designed to highlight hidden information, inference, and deception.

* * *

3. Architectural Anchor: The 4×14×n Tensor (with a 4×13×n Base)
----------------------------------------------------------------

The **only locked-in technical commitment** is a **4×14×n card tensor** as the _canonical_ internal view of “what’s going on in the deck,” with a guaranteed **4×13×n base slice** corresponding to the standard 52-card deck.

### 3.1 The Invariant Card Grid

* **4 rows:** suits (♠ ♥ ♦ ♣) in a fixed, global order.

* **14 columns:** ranks in a fixed, global order:
  
  * Columns 0–12: the standard ranks (2–10, J, Q, K, A).
  
  * Column 13: a dedicated **JOKER / special rank** column.

* Each position (row, column 0–12) uniquely identifies a standard card in a 52-card deck.
* Positions in column 13 identify **joker or special cards**; different rulesets may choose how many of these cells are actually used.

This grid is **game-agnostic**; it doesn’t care if we’re playing Texas Hold’em, Spades, Hearts, or something new. It simply ensures that:

* For **52-card games**, the joker column (column 13) is guaranteed to be all zeros.
* For **54-card decks** (with two jokers), a small, documented subset of joker cells (e.g., “red joker” and “black joker”) are used; the rest remain zero.
* More exotic games can, in principle, use additional joker/special cells, but must do so via explicit, documented invariants.

### 3.2 The Flexible n Channels

Along the third dimension, UNGAR maintains **n channels** that encode _how_ each card currently participates in the game from a given perspective. Examples (non-exhaustive, not all required at once):

* Ownership & location:
  
  * “in my hand”, “in opponent k’s hand”, “in the stock”, “on the board/trick”, “in discard/history”, “in meld”.

* Visibility:
  
  * “publicly visible”, “known to me but not public”, “unknown to me”.

* Status flags:
  
  * “played this trick”, “candidate legal move”, “forbidden by rules”.

* Game-specific context:
  
  * In Poker: involvement in current pot, equity heuristics, position (BTN, SB, BB).
  
  * In Spades/Hearts: void suits, trump status, penalty suits, safe vs risky leads.
  
  * In Gin: part of potential melds, deadwood value, knock eligibility, etc.

The joker column participates in these channels just like any other rank: a joker being in hand, on the table, or in the stock is represented by the same ownership/visibility/status features as non-joker cards.

### 3.3 The Canonical 4×13×n Base Slice

Even though the full tensor is 4×14×n, UNGAR treats the **leftmost 4×13×n slice** (standard ranks only) as the **canonical base**:

* All pure 52-card games live entirely within this base slice; the joker column is all zeros.
* Any model or tool designed for 4×13×n can be extended to 4×14×n by:
  
  * Zero-padding the joker column for 52-card rulesets, or
  
  * Learning to treat the joker column as an optional extension.

Crucially:

> **All downstream intelligence—RL agents, heuristic evaluators, explainability overlays—must treat this 4×14×n tensor (with its 4×13×n base) as the primary lens on the game.**

The project encourages—but does not require—downstream models to treat this tensor somewhat analogously to an image, enabling spatial pattern learning (e.g., flushes, sequences, distributions) and cross-game generalization.

* * *

4. What UNGAR Should Strive to Become
-------------------------------------

This section is written in “should” language on purpose: it’s an aspirational spec for the LLM and future humans.

### 4.1 A General Card Game Substrate

UNGAR should:

* Represent **a wide range of card games**—poker variants, trick-taking games, rummy-style games, shedding games—without changing the core representation.

* Allow game designers and researchers to **define new games declaratively** (as rule configurations over the same 4×14×n substrate).

* Support **multi-player, team, and partnership** structures (e.g., Bridge, Spades partnerships) as first-class concepts.([openspiel.readthedocs.io](https://openspiel.readthedocs.io/en/latest/games.html?utm_source=chatgpt.com "Available games - OpenSpiel documentation"))

### 4.2 A Unified Experiment Playground

UNGAR should serve as a **playground for algorithms**, not just a set of environments:

* Make it easy to plug in RL, search, CFR, evolutionary, and heuristic agents.

* Support both **self-play** (DeepStack/Pluribus-style training) and **cross-play** (different agents and style matchups).([arXiv](https://arxiv.org/abs/1701.01724?utm_source=chatgpt.com "DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"))

* Provide standardized **benchmarks and metrics** across games:
  
  * Winrates, exploitability proxies, convergence curves, style diversity.

The aspirational outcome: **a single experimental stack** where new card AI ideas can be tested across many games, using consistent tools.

### 4.3 A Hub for Transfer and “General Card Intelligence”

UNGAR should explicitly encourage **cross-game generalization experiments**, such as:

* Train an agent on 1–2 games, then test zero- or few-shot on a _new_ game.

* Fine-tune a Poker-trained model on Spades and observe how quickly it learns trick-taking principles.

* Investigate whether features the model learns (e.g., “dangerous suits”, “positional advantage”) show up consistently across multiple games.

The long-term, “moonshot” vision:

> **A single “General Card Player” model that starts competent in multiple games and can adapt to new rule sets with minimal additional training.**

### 4.4 Explainable, Visual, and Human-Legible

UNGAR is **XAI-first** by design:

* The 4×14 grid naturally lends itself to **heatmap overlays**:
  
  * Which cards or suits were most influential in the agent’s decision?
  
  * Which hidden cards (including jokers, if present) did the agent act as if it “believed” were likely?

* Agents should be instrumented to produce:
  
  * **Per-decision attributions** over the 4×14 grid.
  
  * Simple, structured explanations (e.g., “I avoided leading hearts because they carry penalty risk”).

* The platform should make it easy to build:
  
  * Interactive replays of hands with overlays.
  
  * Comparative visualizations of different agents’ style (e.g., tight vs loose, cautious vs aggressive).

The goal is not only to **beat baselines**, but to **illuminate card strategy** in a way that’s accessible to students, researchers, and even serious hobby players.

### 4.5 An Open, Extensible Research Commons

UNGAR aspires to be an **open commons** for imperfect-information card research, similar in spirit to RLCard/OpenSpiel but:

* With a **stronger focus on cross-game representation** (via 4×14×n, with a 4×13×n base).

* With **community-driven game definitions** contributed as rule sets over the shared substrate.

* With clear hooks for:
  
  * Benchmark contributions (new tasks, challenge problems).
  
  * Agent contributions (reference bots, teaching bots, human-like style bots).
  
  * Curriculum definitions (training on game sequences or mixtures).

The project should be architected so that **new games and agents are easy to add without modifying the core**.

* * *

5. Relationship to Other Projects (Context, Not Requirements)
-------------------------------------------------------------

This section is to help an LLM understand the landscape; nothing here is prescriptive.

* **RLCard** is a Python toolkit focusing on multiple card games with unified interfaces and flexible state/action encodings. It shows there is strong demand for such a platform. UNGAR aims to push further toward _one canonical representation_ across games.([IJCAI](https://www.ijcai.org/proceedings/2020/0764.pdf?utm_source=chatgpt.com "RLCard: A Platform for Reinforcement Learning in Card ..."))

* **OpenSpiel** is a general RL/game framework supporting many game types (including card games like Gin Rummy and Bridge variants), with a strong emphasis on algorithms and reproducible research. UNGAR fits beside it as a **card-domain-specialized layer** with a single shared tensor backbone.([arXiv](https://arxiv.org/pdf/1908.09453?utm_source=chatgpt.com "A Framework for Reinforcement Learning in Games"))

* **DeepStack** and **Pluribus** are landmark poker AIs. UNGAR doesn’t attempt to replicate their exact algorithms, but strives to be a **fertile testbed** where DeepStack-style or Pluribus-style ideas can be explored in _many_ games, not only No-Limit Hold’em.([arXiv](https://arxiv.org/abs/1701.01724?utm_source=chatgpt.com "DeepStack: Expert-Level Artificial Intelligence in No-Limit Poker"))

* * *

6. Guiding Principles for Any Implementation
--------------------------------------------

When you later ask an LLM to design architecture, write code, or propose milestones, it should stay aligned with these **high-level principles**:

1. **4×14×n is canonical; 4×13×n is the base.**
   
   * All game states and explainability artifacts should be able to flow through the 4×14×n representation.
   
   * The leftmost 4×13×n slice corresponds to the standard 52-card deck and remains stable across rulesets.
   
   * The joker/special column is reserved, explicitly documented, and may be all zeros for many games.

2. **Card physics over ad-hoc scripts.**
   
   * New games are modeled as rule configurations over shared card/zone/action primitives.

3. **Cross-game thinking.**
   
   * Wherever possible, choose designs that make it easier to reuse agents, models, and analysis tools across multiple games.

4. **XAI and visualization are first-class outcomes.**
   
   * A “good result” is not only higher winrate, but also more insight into the game and the agent’s reasoning.

5. **Open, documented, and community-friendly.**
   
   * Favor clarity and extensibility over hyper-optimized, opaque solutions.
   
   * Make it easy for others to plug in games, agents, and experiments.
```
