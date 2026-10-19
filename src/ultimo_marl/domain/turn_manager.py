"""Turn progression and direction management matching official Ultimo rules."""

from __future__ import annotations

from enum import IntEnum, unique


@unique
class Direction(IntEnum):
    """Cyclic rotation direction around the table."""

    CLOCKWISE = 1
    COUNTER_CLOCKWISE = -1

    def toggle(self) -> Direction:
        """Flip the play direction."""
        return Direction.COUNTER_CLOCKWISE if self == Direction.CLOCKWISE else Direction.CLOCKWISE


class TurnManager:
    """Manages cyclic player rotation, skips, direction reversals, and defense states."""

    __slots__ = (
        "_current_index",
        "_direction",
        "_num_players",
        "finished_indices",
        "pending_draw_penalty",
        "pending_skip_deflection",
    )

    def __init__(self, num_players: int) -> None:
        if num_players < 2:
            raise ValueError(f"At least 2 players are required, got {num_players}")
        self._num_players = num_players
        self._current_index = 0
        self._direction = Direction.CLOCKWISE
        self.pending_skip_deflection: bool = False
        self.pending_draw_penalty: int = 0
        self.finished_indices: set[int] = set()

    @property
    def current_index(self) -> int:
        """0-indexed index of the active player."""
        return self._current_index

    @property
    def direction(self) -> Direction:
        """Current rotation direction."""
        return self._direction

    @property
    def num_players(self) -> int:
        """Total number of players in the game."""
        return self._num_players

    @property
    def active_player_count(self) -> int:
        """Number of active players currently remaining holding cards."""
        return self._num_players - len(self.finished_indices)

    def mark_player_finished(self, index: int) -> None:
        """Mark a player as finished (podium placement claimed)."""
        self.finished_indices.add(index)

    def is_player_finished(self, index: int) -> bool:
        """Return True if specified player has finished and exited turn rotation."""
        return index in self.finished_indices

    @property
    def is_two_player_game(self) -> bool:
        """Return True if match is a 2-player duel (or exactly 2 active players remain)."""
        return self.active_player_count == 2

    def peek_next_player(self, steps: int = 1) -> int:
        """Return the player index N steps ahead without advancing the turn."""
        return self._calculate_index(steps)

    def advance_turn(self, steps: int = 1) -> int:
        """Advance the turn by N steps in current direction. Return new active index."""
        self._current_index = self._calculate_index(steps)
        return self._current_index

    def skip_turn(self) -> int:
        """Skip the next player in order (advancing 2 steps)."""
        return self.advance_turn(2)

    def reverse_direction(self) -> None:
        """Reverse turn direction.

        Note: In official Ultimo rules, in a 2-player game Reverse acts as a Skip.
        """
        if self.is_two_player_game:
            self.skip_turn()
        else:
            self._direction = self._direction.toggle()

    def set_current_index(self, index: int) -> None:
        """Set the active player index directly (used on match initialization)."""
        if not (0 <= index < self._num_players):
            raise IndexError(f"Index {index} out of bounds for {self._num_players} players")
        self._current_index = index

    def _calculate_index(self, steps: int) -> int:
        if self._num_players == 0:
            return 0
        active_count = self.active_player_count
        if active_count == 0:
            return self._current_index % self._num_players
        if active_count == 1:
            for i in range(self._num_players):
                if i not in self.finished_indices:
                    return i
            return self._current_index % self._num_players

        total = self._num_players
        step_dir = self._direction.value
        curr = self._current_index
        active_steps = 0
        while active_steps < steps:
            curr = (curr + step_dir + total) % total
            if curr not in self.finished_indices:
                active_steps += 1
        return curr
