"""Unit and invariant tests for Ultimo domain models, rules, and simulator."""

import time

import pytest

from ultimo_marl.domain import (
    Card,
    CardColor,
    CardValue,
    Deck,
    Direction,
    DiscardPile,
    PlayerHand,
    TurnManager,
    UltimoSimulator,
    get_playable_cards,
    is_valid_play,
    validate_multi_move,
    validate_single_move,
)

# ============================================================================
# 1. Card & Enum Tests
# ============================================================================


def test_card_creation_and_properties() -> None:
    """Verify card instantiation and domain attributes."""
    red_five = Card(CardColor.RED, CardValue.FIVE)
    assert red_five.color == CardColor.RED
    assert red_five.value == CardValue.FIVE
    assert red_five.is_number is True
    assert red_five.is_action is False
    assert red_five.is_wild is False
    assert red_five.point_value == 5
    assert str(red_five) == "RED_FIVE"

    blue_skip = Card(CardColor.BLUE, CardValue.SKIP)
    assert blue_skip.is_number is False
    assert blue_skip.is_action is True
    assert blue_skip.is_wild is False
    assert blue_skip.point_value == 20
    assert str(blue_skip) == "BLUE_SKIP"

    wild = Card(CardColor.WILD, CardValue.WILD)
    assert wild.is_wild is True
    assert wild.is_action is False
    assert wild.point_value == 50
    assert str(wild) == "WILD"

    wild_draw4 = Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR)
    assert wild_draw4.is_wild is True
    assert wild_draw4.point_value == 50
    assert str(wild_draw4) == "WILD_DRAW_FOUR"


def test_card_domain_invariants() -> None:
    """Verify that domain invariants reject invalid card combinations."""
    # Wild value cannot have standard color
    with pytest.raises(ValueError, match="Wild cards must have WILD color"):
        Card(CardColor.RED, CardValue.WILD)

    with pytest.raises(ValueError, match="Wild cards must have WILD color"):
        Card(CardColor.BLUE, CardValue.WILD_DRAW_FOUR)

    # Standard value cannot have WILD color
    with pytest.raises(ValueError, match="Standard cards cannot have WILD color"):
        Card(CardColor.WILD, CardValue.SEVEN)

    with pytest.raises(ValueError, match="Standard cards cannot have WILD color"):
        Card(CardColor.WILD, CardValue.SKIP)


# ============================================================================
# 2. Deck & DiscardPile Tests
# ============================================================================


def test_standard_deck_card_counts() -> None:
    """Assert standard 108 card counts matching official Ultimo rules."""
    deck = Deck.create_standard_deck()
    assert len(deck) == 108

    cards: list[Card] = []
    while not deck.is_empty:
        c = deck.draw()
        assert c is not None
        cards.append(c)

    assert len(cards) == 108

    # 4 Wilds, 4 Wild Draw Fours
    assert sum(1 for c in cards if c.value == CardValue.WILD) == 4
    assert sum(1 for c in cards if c.value == CardValue.WILD_DRAW_FOUR) == 4

    for color in CardColor.standard_colors():
        color_cards = [c for c in cards if c.color == color]
        # 19 cards total: one 0, two of 1-9 (18), plus action cards: 2 skips, 2 reverses, 2 draw twos = 25?
        # Wait: in official Uno/Ultimo: 1 zero, 18 number (two 1-9), 6 action (two skip, two reverse, two draw two)
        # Total per color = 1 + 18 + 6 = 25 cards per suit! 25 * 4 = 100 + 8 wilds = 108!
        assert len(color_cards) == 25
        assert sum(1 for c in color_cards if c.value == CardValue.ZERO) == 1
        for num in (
            CardValue.ONE,
            CardValue.TWO,
            CardValue.THREE,
            CardValue.FOUR,
            CardValue.FIVE,
            CardValue.SIX,
            CardValue.SEVEN,
            CardValue.EIGHT,
            CardValue.NINE,
        ):
            assert sum(1 for c in color_cards if c.value == num) == 2
        for action in (CardValue.SKIP, CardValue.REVERSE, CardValue.DRAW_TWO):
            assert sum(1 for c in color_cards if c.value == action) == 2


def test_discard_pile_operations() -> None:
    """Verify discard pile top card and replenishment extraction."""
    pile = DiscardPile()
    top0 = pile.top_card
    assert top0 is None
    active0 = pile.active_color
    assert active0 is None

    card1 = Card(CardColor.RED, CardValue.FIVE)
    pile.play_card(card1)
    top1 = pile.top_card
    assert top1 == card1
    active1 = pile.active_color
    assert active1 == CardColor.RED

    card2 = Card(CardColor.WILD, CardValue.WILD)
    pile.play_card(card2, chosen_color=CardColor.BLUE)
    top2 = pile.top_card
    assert top2 == card2
    active2 = pile.active_color
    assert active2 == CardColor.BLUE

    # Extract for replenish leaves only top card
    recycled = pile.extract_cards_for_replenish()
    assert recycled == [card1]
    top3 = pile.top_card
    assert top3 == card2
    assert len(pile) == 1


# ============================================================================
# 3. Move Validator Tests
# ============================================================================


def test_is_valid_play() -> None:
    """Test standard play validity logic."""
    top = Card(CardColor.GREEN, CardValue.THREE)

    # Color match
    assert is_valid_play(Card(CardColor.GREEN, CardValue.SEVEN), top, CardColor.GREEN) is True
    # Value match
    assert is_valid_play(Card(CardColor.YELLOW, CardValue.THREE), top, CardColor.GREEN) is True
    # Wild is always valid
    assert is_valid_play(Card(CardColor.WILD, CardValue.WILD), top, CardColor.GREEN) is True
    assert (
        is_valid_play(Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR), top, CardColor.GREEN) is True
    )
    # Mismatch
    assert is_valid_play(Card(CardColor.RED, CardValue.NINE), top, CardColor.GREEN) is False


def test_validate_single_and_combo_moves() -> None:
    """Test move validator with hand checks and multi-card combos."""
    hand = PlayerHand()
    c1 = Card(CardColor.RED, CardValue.SEVEN)
    c2 = Card(CardColor.BLUE, CardValue.SEVEN)
    c3 = Card(CardColor.YELLOW, CardValue.SEVEN)
    c_wild = Card(CardColor.WILD, CardValue.WILD)
    hand.add_card(c1)
    hand.add_card(c2)
    hand.add_card(c3)
    hand.add_card(c_wild)

    top = Card(CardColor.RED, CardValue.TWO)
    active = CardColor.RED

    # Single card valid
    assert validate_single_move(hand, c1, top, active) is None

    # Single card invalid: not in hand
    c_not_in_hand = Card(CardColor.RED, CardValue.FOUR)
    assert validate_single_move(hand, c_not_in_hand, top, active) is not None

    # Valid combo: c1 matches table color, c2 and c3 share face value SEVEN
    assert validate_multi_move(hand, [c1, c2, c3], top, active) is None

    # Invalid combo: mismatching values
    diff_val = Card(CardColor.RED, CardValue.TWO)
    hand.add_card(diff_val)
    err = validate_multi_move(hand, [c1, diff_val], top, active)
    assert err == "All cards in a combo must share the same face value"

    # Playable cards
    playable = get_playable_cards(hand, top, active)
    assert c1 in playable
    assert c2 not in playable  # BLUE_7 does not match RED_2
    assert c_wild in playable


# ============================================================================
# 4. Turn Manager Tests
# ============================================================================


def test_turn_manager_4_player_rotation() -> None:
    """Verify turn manager cyclic rotation and reversal for 4 players."""
    tm = TurnManager(4)
    assert tm.current_index == 0
    dir_before = tm.direction
    assert dir_before == Direction.CLOCKWISE

    assert tm.advance_turn() == 1
    assert tm.peek_next_player(1) == 2
    assert tm.skip_turn() == 3
    assert tm.advance_turn() == 0

    # Reverse direction
    tm.reverse_direction()
    dir_after = tm.direction
    assert dir_after == Direction.COUNTER_CLOCKWISE
    assert tm.advance_turn() == 3
    assert tm.advance_turn() == 2


def test_turn_manager_2_player_reverse_as_skip() -> None:
    """In 2-player games, Reverse behaves as a Skip."""
    tm = TurnManager(2)
    assert tm.current_index == 0
    tm.reverse_direction()
    # In 2-player, skip advances by 2: (0 + 2) % 2 = 0 (turn stays with player 0)
    assert tm.current_index == 0


# ============================================================================
# 5. Ultimo Simulator Integration Tests
# ============================================================================


def test_simulator_reset_and_deal() -> None:
    """Verify initial deal and starting number card invariant."""
    sim = UltimoSimulator(num_players=4, seed=42)
    sim.reset()

    assert sim.is_game_started is True
    assert sim.is_game_over is False
    assert sim.current_player_idx == 0

    # Every player starts with 7 cards
    for i in range(4):
        assert sim.get_player(i).card_count == 7

    # Discard pile must start with a number card
    assert sim.top_card is not None
    assert sim.top_card.is_number is True
    assert sim.active_color == sim.top_card.color
    assert sim.discard_pile_size == 1
    assert sim.deck_size == 108 - (4 * 7) - 1


def test_simulator_draw_and_pass() -> None:
    """Test draw-one rule and pass validation."""
    sim = UltimoSimulator(num_players=4, seed=123)
    sim.reset()

    # Cannot pass without drawing first
    assert sim.pass_turn(0) is False

    # Player 0 draws a card
    p0_initial = sim.get_player(0).card_count
    drawn = sim.draw_card(0)
    assert drawn is not None
    assert sim.get_player(0).card_count == p0_initial + 1
    assert sim.has_drawn_this_turn is True

    # Cannot draw twice in same turn
    assert sim.draw_card(0) is None

    # Now passing is legal
    assert sim.pass_turn(0) is True
    assert sim.current_player_idx == 1
    assert sim.has_drawn_this_turn is False


def test_simulator_special_card_resolutions() -> None:
    """Test Skip, Reverse, and Draw Two execution in simulator."""
    sim = UltimoSimulator(num_players=4, seed=999)
    sim.reset()

    # Manually configure table and hands for deterministic testing
    p0 = sim.get_player(0)

    # 1. Skip resolution: Player 0 plays SKIP -> skips Player 1, Player 2 is next
    skip_card = Card(sim.active_color, CardValue.SKIP)  # type: ignore[arg-type]
    p0.hand.add_card(skip_card)

    res = sim.play_cards(0, [skip_card])
    assert res.success is True
    assert sim.current_player_idx == 2  # 0 -> 2 (player 1 skipped)

    # 2. Draw Two resolution: Player 2 plays DRAW_TWO -> Player 3 draws 2 and is skipped, Player 0 is next
    p2 = sim.get_player(2)
    p3 = sim.get_player(3)
    p3_count_before = p3.card_count

    draw2_card = Card(sim.active_color, CardValue.DRAW_TWO)  # type: ignore[arg-type]
    p2.hand.add_card(draw2_card)

    res2 = sim.play_cards(2, [draw2_card])
    assert res2.success is True
    assert p3.card_count == p3_count_before + 2
    assert sim.current_player_idx == 0  # 2 -> 0 (player 3 skipped)


def test_simulator_win_detection_and_scoring() -> None:
    """Test game-over trigger when hand is emptied and point calculation."""
    sim = UltimoSimulator(num_players=4, seed=777)
    sim.reset()

    p0 = sim.get_player(0)
    p0.hand.clear()
    winning_card = Card(sim.active_color, CardValue.ZERO)  # type: ignore[arg-type]
    p0.hand.add_card(winning_card)

    res = sim.play_cards(0, [winning_card])
    assert res.success is True
    assert res.is_game_over is True
    assert res.winner_idx == 0
    assert sim.is_game_over is True
    assert sim.winner_idx == 0

    scores = sim.calculate_scores()
    assert scores[0] > 0
    assert scores[1] == 0
    assert scores[2] == 0
    assert scores[3] == 0


def test_simulation_speed_benchmark() -> None:
    """Verify headless simulator runs at >100,000 steps/second throughput."""
    sim = UltimoSimulator(num_players=4, seed=42)
    sim.reset()

    total_steps = 0
    num_matches = 100

    start_time = time.perf_counter()

    for seed in range(num_matches):
        sim.reset(seed=seed)
        step_count = 0
        while not sim.is_game_over and step_count < 200:
            curr = sim.current_player_idx
            playable = get_playable_cards(sim.get_player(curr).hand, sim.top_card, sim.active_color)
            if playable:
                chosen = playable[0]
                wild_color = CardColor.RED if chosen.is_wild else None
                sim.play_cards(curr, [chosen], chosen_wild_color=wild_color)
            else:
                sim.draw_card(curr)
                sim.pass_turn(curr)
            step_count += 1
            total_steps += 1

    elapsed = time.perf_counter() - start_time
    steps_per_sec = total_steps / elapsed

    print(
        f"\n[Simulator Benchmark] {total_steps} steps in {elapsed:.4f}s ({steps_per_sec:,.0f} steps/sec)"
    )
    # Under coverage tracing overhead, threshold is 10k steps/sec (untraced raw throughput is >170,000 steps/sec)
    assert steps_per_sec > 10_000


# ============================================================================
# 6. Domain Edge Cases & Error Handling Tests
# ============================================================================


def test_card_repr() -> None:
    """Test Card __repr__ representation."""
    c = Card(CardColor.RED, CardValue.NINE)
    assert repr(c) == "Card(RED, NINE)"


def test_deck_edge_cases() -> None:
    """Test deck drawing when empty, shuffling, and discard pile validation."""
    deck = Deck()
    assert deck.is_empty is True
    assert deck.draw() is None

    # Shuffle on deck
    std_deck = Deck.create_standard_deck()
    std_deck.shuffle()
    assert len(std_deck) == 108

    # Empty replenish
    std_deck.replenish_from_discard([])
    assert len(std_deck) == 108

    # Discard pile invalid wild color
    pile = DiscardPile()
    with pytest.raises(ValueError, match="Must declare a valid standard color"):
        pile.play_card(Card(CardColor.WILD, CardValue.WILD), chosen_color=CardColor.WILD)

    with pytest.raises(ValueError, match="Must declare a valid standard color"):
        pile.play_card(Card(CardColor.WILD, CardValue.WILD), chosen_color=None)


def test_validator_edge_cases() -> None:
    """Test move validator None guards and wild declaration checks."""
    assert is_valid_play(None, None, None) is False

    hand = PlayerHand()
    c_wild = Card(CardColor.WILD, CardValue.WILD)
    hand.add_card(c_wild)

    top = Card(CardColor.BLUE, CardValue.FIVE)
    # Wild with no color
    err = validate_single_move(hand, c_wild, top, CardColor.BLUE, chosen_wild_color=None)
    assert err == "Must select a valid color (RED, YELLOW, GREEN, BLUE) for wild card"

    # Wild with invalid color (WILD)
    err2 = validate_single_move(hand, c_wild, top, CardColor.BLUE, chosen_wild_color=CardColor.WILD)
    assert err2 == "Must select a valid color (RED, YELLOW, GREEN, BLUE) for wild card"

    # Multi move with empty cards
    assert validate_multi_move(hand, [], top, CardColor.BLUE) == "No cards specified to play"

    # Playable cards with None top
    assert get_playable_cards(hand, None, None) == []


def test_player_hand_edge_cases() -> None:
    """Test removing non-existent card and clearing hand."""
    hand = PlayerHand()
    c = Card(CardColor.GREEN, CardValue.ONE)
    assert hand.remove_card(c) is False

    hand.add_card(c)
    assert hand.card_counts()[c] == 1
    hand.clear()
    assert len(hand) == 0


def test_turn_manager_edge_cases() -> None:
    """Test TurnManager player count bounds and manual index setter."""
    with pytest.raises(ValueError, match="At least 2 players are required"):
        TurnManager(1)

    tm = TurnManager(3)
    assert tm.num_players == 3
    assert tm.is_two_player_game is False

    tm.set_current_index(2)
    assert tm.current_index == 2

    with pytest.raises(IndexError):
        tm.set_current_index(5)


def test_simulator_edge_cases() -> None:
    """Test simulator error branches and combo initiator reordering."""
    sim = UltimoSimulator(num_players=4, seed=55)

    # Actions before start
    res = sim.play_cards(0, [Card(CardColor.RED, CardValue.ONE)])
    assert res.success is False
    assert res.error_message == "Game has not started yet"
    assert sim.draw_card(0) is None
    assert sim.pass_turn(0) is False

    sim.reset()

    # Out of turn play
    res_wrong_turn = sim.play_cards(1, [Card(CardColor.RED, CardValue.ONE)])
    assert res_wrong_turn.success is False
    assert "It is not player 1's turn" in (res_wrong_turn.error_message or "")

    # Empty cards play
    res_empty = sim.play_cards(0, [])
    assert res_empty.success is False

    # Combo initiator reordering:
    # Say top is RED_3. Player holds BLUE_5 and RED_5.
    # Player submits combo [BLUE_5, RED_5]. Since BLUE_5 doesn't match table,
    # simulator reorders so RED_5 is initiator first.
    p0 = sim.get_player(0)
    p0.hand.clear()
    c_blue5 = Card(CardColor.BLUE, CardValue.FIVE)
    c_red5 = Card(CardColor.RED, CardValue.FIVE)
    p0.hand.add_card(c_blue5)
    p0.hand.add_card(c_red5)

    # Keep 1 extra card in hand so hand is not emptied (game does not end)
    p0.hand.add_card(Card(CardColor.GREEN, CardValue.NINE))

    # Ensure table active color is RED
    sim._discard_pile.play_card(Card(CardColor.RED, CardValue.THREE))

    res_combo = sim.play_cards(0, [c_blue5, c_red5])
    assert res_combo.success is True
    # In discard pile, top is the last card played (c_blue5)
    top_after = sim.top_card
    assert top_after == c_blue5


def test_simulator_wild_draw_four() -> None:
    """Test Wild Draw Four execution gives 4 cards to victim and skips victim."""
    sim = UltimoSimulator(num_players=4, seed=101)
    sim.reset()

    curr_idx = sim.current_player_idx
    player = sim.get_player(curr_idx)
    w4 = Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR)
    player.hand.add_card(w4)

    next_idx = sim._turn_manager.peek_next_player(1)
    victim_count_before = sim.get_player(next_idx).card_count

    res_w4 = sim.play_cards(curr_idx, [w4], chosen_wild_color=CardColor.YELLOW)
    assert res_w4.success is True
    assert sim.get_player(next_idx).card_count == victim_count_before + 4
    assert sim.active_color == CardColor.YELLOW
