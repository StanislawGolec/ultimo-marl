"""Domain-expert rule-based heuristic agent for benchmark evaluation."""

from __future__ import annotations

import numpy as np

from ultimo_marl.agents.base import BaseAgent
from ultimo_marl.env.action_mapping import (
    ACTION_DRAW,
    WILD_DRAW4_START_IDX,
    WILD_START_IDX,
)


class RuleBasedHeuristicAgent(BaseAgent):
    """Domain-expert rule-based agent serving as an advanced benchmark.

    Strategic Heuristics:
    1. Defensive Aggression: If the next opponent has <= 3 cards or is calling 'Ultimo!',
       prioritizes attack cards (DRAW_TWO, WILD_DRAW_FOUR, SKIP) to disrupt their win.
    2. Hand Diversity Maximization: When playing Wilds, declares the suit most heavily
       represented in its current hand.
    3. Wild Conservation: Preserves Wilds for emergencies, preferring standard plays when valid.
    4. Penalty Point Minimization: Prefers dumping high-value 20-point action cards when safe.
    """

    def __init__(self, name: str = "HeuristicAgent") -> None:
        super().__init__(name=name)

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Select an action using domain-expert card game heuristics."""
        legal_actions = [int(a) for a in np.where(action_mask > 0)[0]]
        if not legal_actions:
            return ACTION_DRAW

        if len(legal_actions) == 1:
            return int(legal_actions[0])

        # Categorize legal actions
        standard_actions = [a for a in legal_actions if a < WILD_START_IDX]
        regular_wild_actions = [
            a for a in legal_actions if WILD_START_IDX <= a < WILD_DRAW4_START_IDX
        ]
        w4_actions = [a for a in legal_actions if WILD_DRAW4_START_IDX <= a < ACTION_DRAW]

        # Extract features from observation vector
        # Offset 18..70 is hand histogram (4 suits * 13 values)
        suit_counts = [float(np.sum(obs[18 + s * 13 : 18 + (s + 1) * 13])) for s in range(4)]
        best_suit_idx = int(np.argmax(suit_counts))

        # Offset 72 is normalized card count of next player in clockwise rotation
        next_opp_norm = float(obs[72])
        next_opp_cards = int(round(next_opp_norm * 10.0))
        # Offset 76 is any_opp_ultimo flag (1.0 if any opponent has <= 1 card)
        any_opp_ultimo = float(obs[76]) > 0.5
        next_opp_in_danger = next_opp_cards <= 3 or any_opp_ultimo

        # Group standard actions by action type (value index = action % 13)
        draw2_actions = [a for a in standard_actions if a % 13 == 12]
        skip_actions = [a for a in standard_actions if a % 13 == 10]
        reverse_actions = [a for a in standard_actions if a % 13 == 11]
        number_actions = [a for a in standard_actions if a % 13 <= 9]

        # ---------------------------------------------------------------------
        # STRATEGY 1: Disruption when next opponent is close to winning (<= 3 cards)
        # ---------------------------------------------------------------------
        if next_opp_in_danger:
            # 1a. Draw Two
            if draw2_actions:
                return self._best_suit_standard_play(draw2_actions, best_suit_idx)

            # 1b. Wild Draw Four (declare our most frequent suit)
            target_w4 = WILD_DRAW4_START_IDX + best_suit_idx
            if target_w4 in w4_actions:
                return target_w4
            if w4_actions:
                return int(w4_actions[0])

            # 1c. Skip next player
            if skip_actions:
                return self._best_suit_standard_play(skip_actions, best_suit_idx)

            # 1d. Reverse play direction
            if reverse_actions:
                return self._best_suit_standard_play(reverse_actions, best_suit_idx)

        # ---------------------------------------------------------------------
        # STRATEGY 2: Standard Card Play (Conserve Wilds)
        # ---------------------------------------------------------------------
        if standard_actions:
            # If we hold 20-point action cards, dump them to minimize score penalties
            high_value_actions = draw2_actions + skip_actions + reverse_actions
            if high_value_actions:
                return self._best_suit_standard_play(high_value_actions, best_suit_idx)

            # Sort numbers by highest face value to minimize hand point sum
            sorted_numbers = sorted(number_actions, key=lambda a: a % 13, reverse=True)
            # Prefer numbers matching our best suit
            return self._best_suit_standard_play(sorted_numbers, best_suit_idx)

        # ---------------------------------------------------------------------
        # STRATEGY 3: Emergency Wild Play
        # ---------------------------------------------------------------------
        # Play regular wild declaring best suit
        target_regular_wild = WILD_START_IDX + best_suit_idx
        if target_regular_wild in regular_wild_actions:
            return target_regular_wild
        if regular_wild_actions:
            return int(regular_wild_actions[0])

        # Play wild draw four declaring best suit
        target_w4 = WILD_DRAW4_START_IDX + best_suit_idx
        if target_w4 in w4_actions:
            return target_w4
        if w4_actions:
            return int(w4_actions[0])

        # ---------------------------------------------------------------------
        # STRATEGY 4: Draw Card
        # ---------------------------------------------------------------------
        return ACTION_DRAW

    @staticmethod
    def _best_suit_standard_play(candidate_actions: list[int], best_suit_idx: int) -> int:
        """Pick an action belonging to the agent's most numerous suit if available."""
        for a in candidate_actions:
            suit_idx = a // 13
            if suit_idx == best_suit_idx:
                return a
        return candidate_actions[0]
