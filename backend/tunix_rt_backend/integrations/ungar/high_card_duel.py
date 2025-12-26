"""High Card Duel episode to trace conversion.

This module converts UNGAR High Card Duel game episodes into tunix-rt reasoning traces
for use in training and evaluation workflows. The conversion uses minimal, deterministic
natural language to maintain consistency and reproducibility.

Note:
    This module requires UNGAR to be installed (backend[ungar] extra).
    Functions will raise ImportError if UNGAR is not available.
"""

import logging
from typing import Any

from tunix_rt_backend.schemas.trace import ReasoningTrace, TraceStep

logger = logging.getLogger(__name__)


def generate_high_card_duel_traces(count: int, seed: int | None = None) -> list[ReasoningTrace]:
    """Generate High Card Duel traces from UNGAR episodes.

    Plays N episodes of High Card Duel and converts each into a reasoning trace
    suitable for Tunix training workflows.

    Args:
        count: Number of episodes to generate (1-100)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of ReasoningTrace objects with deterministic structure

    Raises:
        ImportError: If UNGAR is not installed
        ValueError: If count is out of valid range (1-100)

    Example trace structure:
        prompt: "High Card Duel: You have 1 hidden card. Action: reveal."
        steps:
          - "Legal moves: [reveal]"
          - "My hand: AS"
          - "Unseen cards: 51"
          - "Action chosen: reveal"
        final_answer: "reveal"
        metadata:
          source: "ungar"
          game: "high_card_duel"
          seed: 42
          episode_index: 0
          my_card: "AS"
          opponent_card: "KH"
          result: "win"
    """
    # Lazy import to avoid requiring UNGAR at module load time
    # UNGAR is an optional dependency, not available to mypy in default CI
    try:
        from ungar.games.high_card_duel import (  # type: ignore[import-not-found]
            make_high_card_duel_spec,
        )
        from ungar.runner import play_random_episode  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("UNGAR is not installed. Install with: pip install -e '.[ungar]'") from e

    # Validate input
    if count < 1 or count > 100:
        raise ValueError(f"count must be between 1 and 100, got {count}")

    # Generate traces
    traces: list[ReasoningTrace] = []
    game_spec = make_high_card_duel_spec()

    for episode_index in range(count):
        # Generate episode seed (deterministic if seed provided)
        episode_seed = seed + episode_index if seed is not None else None

        # Play one episode
        episode_result = play_random_episode(game_spec, seed=episode_seed)

        # Convert episode to trace
        trace = _convert_episode_to_trace(
            episode_result=episode_result,
            episode_index=episode_index,
            seed=seed,
        )
        traces.append(trace)

    return traces


def _convert_episode_to_trace(
    episode_result: dict[str, Any],
    episode_index: int,
    seed: int | None,
) -> ReasoningTrace:
    """Convert a single High Card Duel episode to a reasoning trace.

    Args:
        episode_result: Episode result from play_random_episode
        episode_index: Index of this episode in the batch
        seed: Original seed (for metadata)

    Returns:
        ReasoningTrace with minimal, deterministic natural language
    """
    # Extract game information from episode_result
    # Note: High Card Duel is extremely simple - each player gets 1 card, then reveals
    # The episode_result should contain the game state and outcome

    # For High Card Duel, the initial state has my_hand and opponent_hand
    # Since this is a simple game with only "reveal" as an action, we construct
    # a minimal reasoning trace

    # Parse the episode result (structure depends on UNGAR's play_random_episode output)
    # UNGAR 0.2+ returns an Episode object, not a dict
    # Expected Episode attributes: states, moves, rewards

    # Handle Episode object (new API)
    if hasattr(episode_result, "states"):
        states = episode_result.states
        initial_state = states[0] if states else None
        # rewards is a tuple/list
        returns = episode_result.rewards if hasattr(episode_result, "rewards") else [0, 0]
    else:
        # Handle legacy dict format (fallback)
        initial_state = episode_result.get("initial_state")
        returns = episode_result.get("returns", [0, 0])

    # Extract card information (this is game-specific logic)
    # High Card Duel: each player has exactly 1 card
    my_card_str = _extract_my_card(initial_state)
    opponent_card_str = _extract_opponent_card(episode_result)

    # Determine result
    # rewards is often a tuple of (p1_reward, p2_reward)
    my_return = returns[0] if returns else 0
    result = "win" if my_return > 0 else ("loss" if my_return < 0 else "tie")

    # Construct trace steps (minimal, deterministic)
    steps = [
        TraceStep(i=0, type="legal_moves", content="Legal moves: [reveal]"),
        TraceStep(i=1, type="observation", content=f"My hand: {my_card_str}"),
        TraceStep(i=2, type="unseen", content="Unseen cards: 51"),
        TraceStep(i=3, type="decision", content="Action chosen: reveal"),
    ]

    # Construct metadata
    metadata = {
        "source": "ungar",
        "game": "high_card_duel",
        "episode_index": episode_index,
        "my_card": my_card_str,
        "opponent_card": opponent_card_str,
        "result": result,
    }
    if seed is not None:
        metadata["seed"] = seed

    # Construct full trace
    return ReasoningTrace(
        trace_version="1.0",
        prompt="High Card Duel: You have 1 hidden card. Action: reveal.",
        final_answer="reveal",
        steps=steps,
        meta=metadata,
    )


def _extract_my_card(state: Any) -> str:
    """Extract my card from game state in short format (e.g., 'AS' for Ace of Spades).

    Args:
        state: UNGAR game state

    Returns:
        Card string in short format (rank + suit, e.g., 'AS', 'KH', '2D')
    """
    # Access the tensor or card representation from UNGAR state
    # High Card Duel stores cards in my_hand plane of the tensor
    # For now, return a placeholder - this will be implemented when UNGAR is available
    try:
        # Get my hand cards from the state (Player 0 perspective)
        my_hand = state.to_tensor(player=0).cards_in_plane("my_hand")
        if my_hand:
            card = list(my_hand)[0]  # Get first (and only) card
            return _format_card(card)
        logger.warning("Failed to extract my_card: my_hand is empty")
        return "??"
    except (AttributeError, KeyError) as e:
        logger.warning(f"Failed to extract my_card: {e}")
        return "??"


def _extract_opponent_card(episode_result: Any) -> str:
    """Extract opponent's card from final state.

    Args:
        episode_result: Complete episode result (Episode object or dict)

    Returns:
        Card string in short format
    """
    try:
        final_state = None

        # Handle Episode object
        if hasattr(episode_result, "states"):
            if episode_result.states:
                final_state = episode_result.states[-1]
        # Handle legacy dict
        elif isinstance(episode_result, dict):
            final_state = episode_result.get("final_state")

        if final_state is None:
            logger.warning("Failed to extract opponent_card: final_state is None")
            return "??"

        # In High Card Duel, opponent's card is revealed in final state
        # We view from Player 0's perspective
        opponent_hand = final_state.to_tensor(player=0).cards_in_plane("opponent_hand")
        if opponent_hand:
            card = list(opponent_hand)[0]
            return _format_card(card)
        logger.warning("Failed to extract opponent_card: opponent_hand is empty")
        return "??"
    except (AttributeError, KeyError) as e:
        logger.warning(f"Failed to extract opponent_card: {e}")
        return "??"


def _format_card(card: Any) -> str:
    """Format UNGAR card object to short string (e.g., 'AS', 'KH').

    Args:
        card: UNGAR Card object

    Returns:
        Short string representation (rank + suit)
    """
    try:
        # UNGAR cards have rank and suit enums
        rank_map = {
            "ACE": "A",
            "TWO": "2",
            "THREE": "3",
            "FOUR": "4",
            "FIVE": "5",
            "SIX": "6",
            "SEVEN": "7",
            "EIGHT": "8",
            "NINE": "9",
            "TEN": "T",
            "JACK": "J",
            "QUEEN": "Q",
            "KING": "K",
        }
        suit_map = {
            "SPADES": "S",
            "HEARTS": "H",
            "DIAMONDS": "D",
            "CLUBS": "C",
        }

        rank_str = rank_map.get(card.rank.name, "?")
        suit_str = suit_map.get(card.suit.name, "?")
        return f"{rank_str}{suit_str}"
    except (AttributeError, KeyError) as e:
        logger.warning(f"Failed to format card: {e}")
        return "??"
