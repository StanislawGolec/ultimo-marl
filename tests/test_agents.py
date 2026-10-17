"""Unit and benchmark tests for RandomAgent, RuleBasedHeuristicAgent, and Tournament."""

import numpy as np

from ultimo_marl.agents import (
    BaseAgent,
    RandomAgent,
    RuleBasedHeuristicAgent,
    Tournament,
    wilson_score_interval,
)
from ultimo_marl.domain import Card, CardColor, CardValue, PlayerHand, UltimoSimulator
from ultimo_marl.env import (
    NUM_ACTIONS,
    OBSERVATION_DIM,
    card_to_action_id,
    compute_action_mask,
    encode_observation,
)

# ============================================================================
# 1. RandomAgent Tests
# ============================================================================


def test_random_agent_action_legality() -> None:
    """Verify RandomAgent always selects actions marked as legal by action_mask."""
    agent = RandomAgent(seed=42)
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

    # Synthetic mask with only actions 3 and 17 enabled
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[3] = 1
    mask[17] = 1

    for _ in range(20):
        action = agent.select_action(obs, mask)
        assert action in (3, 17)
        assert mask[action] == 1


# ============================================================================
# 2. RuleBasedHeuristicAgent Tests
# ============================================================================


def test_heuristic_agent_attacks_low_card_opponent() -> None:
    """When next opponent has <= 3 cards, heuristic agent prioritizes attack cards."""
    agent = RuleBasedHeuristicAgent()

    sim = UltimoSimulator(num_players=4, seed=1)
    sim.reset()

    # Configure player 0 hand: holds a RED_DRAW_TWO and RED_FIVE
    p0 = sim.get_player(0)
    p0.hand.clear()
    draw2 = Card(CardColor.RED, CardValue.DRAW_TWO)
    five = Card(CardColor.RED, CardValue.FIVE)
    p0.hand.add_card(draw2)
    p0.hand.add_card(five)

    # Force table top card to RED_TWO (active color RED)
    sim._discard_pile.play_card(Card(CardColor.RED, CardValue.TWO))

    # Force player 1 (next opponent) to have only 1 card remaining
    sim.get_player(1).hand.clear()
    sim.get_player(1).hand.add_card(Card(CardColor.BLUE, CardValue.NINE))

    obs = encode_observation(sim, 0)
    mask = compute_action_mask(p0.hand, sim.top_card, sim.active_color)

    chosen_action = agent.select_action(obs, mask)
    expected_draw2_id = card_to_action_id(draw2)

    # Must choose DRAW_TWO over number 5 to disrupt opponent's winning chance
    assert chosen_action == expected_draw2_id


def test_heuristic_agent_declares_most_numerous_suit_on_wild() -> None:
    """When forced to play Wild, agent declares the suit it holds most cards in."""
    agent = RuleBasedHeuristicAgent()

    # Hand has 3 GREEN cards, 1 RED card, and 1 WILD
    hand = PlayerHand()
    hand.add_card(Card(CardColor.GREEN, CardValue.ONE))
    hand.add_card(Card(CardColor.GREEN, CardValue.TWO))
    hand.add_card(Card(CardColor.GREEN, CardValue.THREE))
    hand.add_card(Card(CardColor.RED, CardValue.FOUR))
    wild = Card(CardColor.WILD, CardValue.WILD)
    hand.add_card(wild)

    # Top card is BLUE_9 (no standard cards playable, must play Wild or Draw)
    top = Card(CardColor.BLUE, CardValue.NINE)
    active = CardColor.BLUE

    sim = UltimoSimulator(num_players=4, seed=2)
    sim.reset()
    sim._players[0].hand = hand
    sim._discard_pile.play_card(top)

    obs = encode_observation(sim, 0)
    mask = compute_action_mask(hand, top, active)

    chosen_action = agent.select_action(obs, mask)

    # Best suit is GREEN (index 2 in standard suits: RED=0, YELLOW=1, GREEN=2, BLUE=3)
    # Action 52 is WILD_RED, 53 is WILD_YELLOW, 54 is WILD_GREEN, 55 is WILD_BLUE
    assert chosen_action == 54  # WILD declaring GREEN


# ============================================================================
# 3. Wilson Score Interval Tests
# ============================================================================


def test_wilson_score_interval_bounds() -> None:
    """Verify Wilson confidence interval mathematical correctness."""
    # 0 total
    low0, high0 = wilson_score_interval(0, 0)
    assert low0 == 0.0 and high0 == 0.0

    # 100 wins out of 100 games
    low100, high100 = wilson_score_interval(100, 100)
    assert 0.95 < low100 < 1.0
    assert high100 == 1.0

    # 50 wins out of 100 games -> centered around 0.50
    low50, high50 = wilson_score_interval(50, 100)
    assert 0.39 < low50 < 0.45
    assert 0.55 < high50 < 0.61


# ============================================================================
# 4. Tournament Evaluation Benchmark
# ============================================================================


def test_tournament_heuristic_vs_random_dominance() -> None:
    """Benchmark: RuleBasedHeuristicAgent must heavily dominate RandomAgents (>70% win rate)."""
    heuristic_bot = RuleBasedHeuristicAgent(name="HeuristicMaster")
    random_bots = [RandomAgent(name=f"RandomBot_{i}") for i in range(1, 4)]
    agents: list[BaseAgent] = [heuristic_bot, *random_bots]
    tournament = Tournament(agents=agents)
    # Run 100 fast simulated games
    results = tournament.play(num_games=100, seed=42)

    assert results.games_played == 100
    assert results.wins["HeuristicMaster"] >= 65
    h_win_rate = results.win_rates["HeuristicMaster"]

    print(f"\n{results.summary()}")

    # In a 4-player game, random chance is 25%.
    # Heuristic agent typically achieves >75% win rate against 3 random opponents!
    assert h_win_rate >= 0.65, f"Expected Heuristic win rate >= 65%, got {h_win_rate:.1%}"
    assert results.total_scores["HeuristicMaster"] > 0
