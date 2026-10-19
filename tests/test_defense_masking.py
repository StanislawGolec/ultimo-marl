"""Unit and invariant tests for tactical Skip deflection, Draw penalty stacking, and action masking."""

from __future__ import annotations

import numpy as np
import torch

from ultimo_marl.domain import (
    Card,
    CardColor,
    CardValue,
    PlayerHand,
    UltimoSimulator,
)
from ultimo_marl.env import (
    ACTION_DRAW,
    NUM_ACTIONS,
    WILD_DRAW4_START_IDX,
    WILD_START_IDX,
    UltimoAECEnv,
    card_to_action_id,
    compute_action_mask,
    encode_observation,
)
from ultimo_marl.training import DuelingQNetwork, mask_q_values


def test_skip_deflection_action_mask_strictness() -> None:
    """When pending_skip_deflection is active, only Skip cards and ACTION_DRAW are legal."""
    hand = PlayerHand(
        [
            Card(CardColor.RED, CardValue.SKIP),
            Card(CardColor.RED, CardValue.FIVE),
            Card(CardColor.BLUE, CardValue.REVERSE),
            Card(CardColor.GREEN, CardValue.DRAW_TWO),
            Card(CardColor.WILD, CardValue.WILD),
            Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR),
        ]
    )
    top_card = Card(CardColor.RED, CardValue.SKIP)
    active_color = CardColor.RED

    mask = compute_action_mask(
        hand=hand,
        top_card=top_card,
        active_color=active_color,
        pending_skip_deflection=True,
        pending_draw_penalty=0,
    )

    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == np.int8

    red_skip_id = card_to_action_id(Card(CardColor.RED, CardValue.SKIP))
    assert mask[red_skip_id] == 1
    assert mask[ACTION_DRAW] == 1

    # Exactly 2 actions are legal (RED_SKIP and ACTION_DRAW); all other 59 actions are 0
    assert int(np.sum(mask)) == 2
    for action_id in range(NUM_ACTIONS):
        if action_id not in (red_skip_id, ACTION_DRAW):
            assert mask[action_id] == 0

    # Verify Q-value logit suppression (-1e9) for all 59 illegal actions
    net = DuelingQNetwork(hidden_dim=64)
    obs = torch.rand(1, 90)
    raw_q = net(obs)
    masked_q = mask_q_values(raw_q, torch.from_numpy(mask).unsqueeze(0))

    for action_id in range(NUM_ACTIONS):
        if action_id in (red_skip_id, ACTION_DRAW):
            assert masked_q[0, action_id].item() > -1e8
        else:
            assert masked_q[0, action_id].item() <= -1e9


def test_draw_stacking_action_mask_strictness() -> None:
    """When pending_draw_penalty > 0, only Draw Two, Wild Draw Four, and ACTION_DRAW are legal."""
    hand = PlayerHand(
        [
            Card(CardColor.RED, CardValue.FIVE),
            Card(CardColor.RED, CardValue.SKIP),
            Card(CardColor.BLUE, CardValue.DRAW_TWO),
            Card(CardColor.WILD, CardValue.WILD),
            Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR),
        ]
    )
    top_card = Card(CardColor.RED, CardValue.DRAW_TWO)
    active_color = CardColor.RED

    mask = compute_action_mask(
        hand=hand,
        top_card=top_card,
        active_color=active_color,
        pending_skip_deflection=False,
        pending_draw_penalty=2,
    )

    blue_d2_id = card_to_action_id(Card(CardColor.BLUE, CardValue.DRAW_TWO))
    expected_legal = {
        blue_d2_id,
        WILD_DRAW4_START_IDX,
        WILD_DRAW4_START_IDX + 1,
        WILD_DRAW4_START_IDX + 2,
        WILD_DRAW4_START_IDX + 3,
        ACTION_DRAW,
    }

    for action_id in range(NUM_ACTIONS):
        if action_id in expected_legal:
            assert mask[action_id] == 1
        else:
            assert mask[action_id] == 0

    # Regular Wild (52..55), RED_5, and RED_SKIP must be masked out (0)
    for w_idx in range(WILD_START_IDX, WILD_DRAW4_START_IDX):
        assert mask[w_idx] == 0

    masked_q = mask_q_values(
        torch.zeros(1, NUM_ACTIONS),
        torch.from_numpy(mask).unsqueeze(0),
    )
    for action_id in range(NUM_ACTIONS):
        if action_id not in expected_legal:
            assert masked_q[0, action_id].item() <= -1e9


def test_simulator_skip_deflection_chain() -> None:
    """Verify Skip deflection passes the skip along without skipping the deflecting player."""
    sim = UltimoSimulator(num_players=4, seed=42)
    sim.reset()

    active_color = sim.active_color
    assert active_color is not None

    # Player 0 plays a Skip card
    p0_skip = Card(active_color, CardValue.SKIP)
    sim.get_player(0).hand.add_card(p0_skip)
    res0 = sim.play_cards(0, [p0_skip])
    assert res0.success is True
    assert bool(sim.pending_skip_deflection) is True
    assert int(sim.current_player_idx) == 1

    # Player 1 attempts to play a regular number card -> rejected while skip is pending
    p1_num = Card(active_color, CardValue.THREE)
    sim.get_player(1).hand.add_card(p1_num)
    res_illegal = sim.play_cards(1, [p1_num])
    assert res_illegal.success is False
    assert int(sim.current_player_idx) == 1
    assert bool(sim.pending_skip_deflection) is True

    # Player 1 plays a Skip card (BLUE_SKIP) to deflect to Player 2
    p1_skip = Card(CardColor.BLUE, CardValue.SKIP)
    sim.get_player(1).hand.add_card(p1_skip)
    res1 = sim.play_cards(1, [p1_skip])
    assert res1.success is True
    assert bool(sim.pending_skip_deflection) is True
    assert int(sim.current_player_idx) == 2
    assert sim.active_color == CardColor.BLUE

    # Player 2 does not deflect and accepts the skip via draw_card -> forfeits turn to Player 3
    p2_cards_before = sim.get_player(2).card_count
    sim.draw_card(2)
    assert bool(sim.pending_skip_deflection) is False
    assert int(sim.current_player_idx) == 3
    assert sim.get_player(2).card_count == p2_cards_before


def test_simulator_draw_penalty_stacking_and_forfeit() -> None:
    """Verify Draw Two and Wild Draw Four penalties stack and forfeit turn upon drawing."""
    sim = UltimoSimulator(num_players=4, seed=99)
    sim.reset()

    active_color = sim.active_color
    assert active_color is not None

    # Player 0 plays DRAW_TWO (+2 penalty to Player 1)
    d2_p0 = Card(active_color, CardValue.DRAW_TWO)
    sim.get_player(0).hand.add_card(d2_p0)
    res0 = sim.play_cards(0, [d2_p0])
    assert res0.success is True
    assert int(sim.pending_draw_penalty) == 2
    assert int(sim.current_player_idx) == 1

    # Player 1 attempts to play a number card -> rejected
    num_p1 = Card(active_color, CardValue.SEVEN)
    sim.get_player(1).hand.add_card(num_p1)
    assert sim.play_cards(1, [num_p1]).success is False

    # Player 1 stacks a GREEN_DRAW_TWO (+4 total penalty to Player 2)
    d2_p1 = Card(CardColor.GREEN, CardValue.DRAW_TWO)
    sim.get_player(1).hand.add_card(d2_p1)
    res1 = sim.play_cards(1, [d2_p1])
    assert res1.success is True
    assert int(sim.pending_draw_penalty) == 4
    assert int(sim.current_player_idx) == 2
    assert str(sim.active_color) == "CardColor.GREEN"

    # Player 2 stacks a WILD_DRAW_FOUR (+8 total penalty to Player 3)
    w4_p2 = Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR)
    sim.get_player(2).hand.add_card(w4_p2)
    res2 = sim.play_cards(2, [w4_p2], chosen_wild_color=CardColor.YELLOW)
    assert res2.success is True
    assert int(sim.pending_draw_penalty) == 8
    assert int(sim.current_player_idx) == 3
    assert str(sim.active_color) == "CardColor.YELLOW"

    # Verify observation vector encodes pending_draw_penalty = 8 / 16.0 = 0.5
    obs_p3 = encode_observation(sim, observing_player_idx=3)
    assert obs_p3[88] == 0.0
    assert np.isclose(obs_p3[89], 8.0 / 16.0)

    # Player 3 draws -> receives all 8 penalty cards and forfeits turn to Player 0
    p3_count_before = sim.get_player(3).card_count
    sim.draw_card(3)
    assert int(sim.pending_draw_penalty) == 0
    assert sim.get_player(3).card_count == p3_count_before + 8
    assert int(sim.current_player_idx) == 0


def test_aec_env_defense_mask_integration() -> None:
    """Verify UltimoAECEnv surfaces defense action masks when Skip or Draw stack is active."""
    env = UltimoAECEnv(num_players=4)
    env.reset(seed=15)

    sim = env._sim
    active_color = sim.active_color
    assert active_color is not None

    # Give player_0 a SKIP card and play it via env.step
    skip_card = Card(active_color, CardValue.SKIP)
    sim.get_player(0).hand.add_card(skip_card)
    env._update_infos()

    skip_action = card_to_action_id(skip_card)
    env.step(skip_action)

    # Now player_1 is active and under pending_skip_deflection
    assert env.agent_selection == "player_1"
    obs_p1, _rew, _term, _trunc, info_p1 = env.last()
    assert obs_p1[88] == 1.0
    mask_p1 = info_p1["action_mask"]
    assert mask_p1[ACTION_DRAW] == 1

    # Any legal non-DRAW action in mask_p1 must correspond to a SKIP card
    for act_id in range(ACTION_DRAW):
        if mask_p1[act_id] == 1:
            assert act_id % 13 == 10  # index 10 in STANDARD_VALUES is CardValue.SKIP
