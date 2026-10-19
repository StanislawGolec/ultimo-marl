"""Unit tests for room discovery, access codes, host commands, and CLI room flags."""

from __future__ import annotations

import pytest

import run_live_bot
from ultimo_marl.agents.random_agent import RandomAgent
from ultimo_marl.bot import LiveBot, LiveGameState
from ultimo_marl.bot import run_live_bot as bot_cli
from ultimo_marl.protocol.client import TcpGameClient
from ultimo_marl.protocol.codec import ProtocolCodec
from ultimo_marl.protocol.events import (
    CommandType,
    CreateRoom,
    EventType,
    Join,
    Kick,
    ListRooms,
    ProtocolException,
    RoomList,
    RoomOwnerChanged,
    RoomStatus,
    Start,
    TransferOwnership,
)


def test_encode_decode_create_room_public_and_private() -> None:
    """Verify CREATE_ROOM wire serialization and deserialization for public and private rooms."""
    cmd_pub = CreateRoom(name="MainLobby", is_public=True, access_code=None, max_players=4)
    assert cmd_pub.command_type == CommandType.CREATE_ROOM
    encoded_pub = ProtocolCodec.encode_command(cmd_pub)
    assert encoded_pub == "CREATE_ROOM MainLobby PUBLIC 4\n"
    assert ProtocolCodec.decode_command(encoded_pub) == cmd_pub

    cmd_priv = CreateRoom(
        name="SecretTable",
        is_public=False,
        access_code="XY99",
        max_players=8,
    )
    encoded_priv = ProtocolCodec.encode_command(cmd_priv)
    assert encoded_priv == "CREATE_ROOM SecretTable PRIVATE XY99 8\n"
    assert ProtocolCodec.decode_command(encoded_priv) == cmd_priv

    # Decode without optional max_players
    decoded_no_max = ProtocolCodec.decode_command("CREATE_ROOM QuickRoom PUBLIC\n")
    assert decoded_no_max == CreateRoom(
        name="QuickRoom",
        is_public=True,
        access_code=None,
        max_players=4,
    )

    decoded_priv_code_only = ProtocolCodec.decode_command("CREATE_ROOM VIPRoom PRIVATE CODE\n")
    assert decoded_priv_code_only == CreateRoom(
        name="VIPRoom",
        is_public=False,
        access_code="CODE",
        max_players=4,
    )


def test_decode_create_room_invalid() -> None:
    """Malformed CREATE_ROOM lines must raise ProtocolException."""
    with pytest.raises(ProtocolException, match="CREATE_ROOM requires name and visibility"):
        ProtocolCodec.decode_command("CREATE_ROOM OnlyName\n")

    with pytest.raises(ProtocolException, match="CREATE_ROOM visibility must be PUBLIC or PRIVATE"):
        ProtocolCodec.decode_command("CREATE_ROOM RoomName HIDDEN\n")


def test_encode_decode_list_rooms_kick_transfer_ownership() -> None:
    """Verify LIST_ROOMS, KICK, and TRANSFER_OWNERSHIP commands round-trip cleanly."""
    list_cmd = ListRooms()
    assert list_cmd.command_type == CommandType.LIST_ROOMS
    assert ProtocolCodec.encode_command(list_cmd) == "LIST_ROOMS\n"
    assert ProtocolCodec.decode_command("LIST_ROOMS\n") == list_cmd

    kick_cmd = Kick(target_username="Griefer_01")
    assert kick_cmd.command_type == CommandType.KICK
    assert ProtocolCodec.encode_command(kick_cmd) == "KICK Griefer_01\n"
    assert ProtocolCodec.decode_command("KICK Griefer_01\n") == kick_cmd

    transfer_cmd = TransferOwnership(target_username="NewHost_02")
    assert transfer_cmd.command_type == CommandType.TRANSFER_OWNERSHIP
    assert ProtocolCodec.encode_command(transfer_cmd) == "TRANSFER_OWNERSHIP NewHost_02\n"
    assert ProtocolCodec.decode_command("TRANSFER_OWNERSHIP NewHost_02\n") == transfer_cmd

    with pytest.raises(ProtocolException, match="KICK requires target username"):
        ProtocolCodec.decode_command("KICK\n")

    with pytest.raises(ProtocolException, match="TRANSFER_OWNERSHIP requires target username"):
        ProtocolCodec.decode_command("TRANSFER_OWNERSHIP\n")


def test_encode_decode_extended_join() -> None:
    """Verify extended JOIN command with room_id, access_code, and max_players."""
    cases = [
        (Join(username="Bot1"), "JOIN Bot1\n"),
        (Join(username="Bot1", room_id="room-10"), "JOIN Bot1 room-10\n"),
        (
            Join(username="Bot1", room_id="room-10", access_code="ABCD"),
            "JOIN Bot1 room-10 ABCD\n",
        ),
        (
            Join(username="Bot1", room_id="room-10", access_code="ABCD", max_players=8),
            "JOIN Bot1 room-10 ABCD 8\n",
        ),
        (
            Join(username="Bot1", room_id="room-10", max_players=12),
            "JOIN Bot1 room-10 12\n",
        ),
        (
            Join(username="Bot1", max_players=6),
            "JOIN Bot1 6\n",
        ),
    ]

    for cmd, expected_wire in cases:
        wire = ProtocolCodec.encode_command(cmd)
        assert wire == expected_wire
        decoded = ProtocolCodec.decode_command(wire)
        assert decoded == cmd


def test_encode_decode_room_list_and_owner_changed_events() -> None:
    """Verify ROOM_LIST and ROOM_OWNER_CHANGED server events serialization and parsing."""
    empty_list = RoomList(rooms=[])
    assert empty_list.event_type == EventType.ROOM_LIST
    assert ProtocolCodec.encode_event(empty_list) == "ROOM_LIST 0\n"
    assert ProtocolCodec.decode_event("ROOM_LIST 0\n") == empty_list
    assert ProtocolCodec.decode_event("ROOM_LIST -\n") == empty_list

    rooms_data = [
        {
            "room_id": "room-1",
            "name": "LobbyAlpha",
            "current_players": 2,
            "max_players": 4,
            "status": "WAITING",
        },
        {
            "room_id": "room-2",
            "name": "Tournament",
            "current_players": 6,
            "max_players": 16,
            "status": "PLAYING",
        },
    ]
    room_list_ev = RoomList(rooms=rooms_data)
    encoded = ProtocolCodec.encode_event(room_list_ev)
    assert encoded == "ROOM_LIST 2 room-1:LobbyAlpha:2/4:WAITING room-2:Tournament:6/16:PLAYING\n"
    decoded = ProtocolCodec.decode_event(encoded)
    assert decoded == room_list_ev

    # Also test server format without leading count
    decoded_no_count = ProtocolCodec.decode_event(
        "ROOM_LIST room-1:LobbyAlpha:2/4:WAITING room-2:Tournament:6/16:PLAYING\n"
    )
    assert decoded_no_count == room_list_ev

    owner_ev = RoomOwnerChanged(new_owner_username="HostBot")
    assert owner_ev.event_type == EventType.ROOM_OWNER_CHANGED
    assert owner_ev.new_owner == "HostBot"
    assert ProtocolCodec.encode_event(owner_ev) == "ROOM_OWNER_CHANGED HostBot\n"
    assert ProtocolCodec.decode_event("ROOM_OWNER_CHANGED HostBot\n") == owner_ev

    with pytest.raises(ProtocolException, match="Malformed ROOM_OWNER_CHANGED"):
        ProtocolCodec.decode_event("ROOM_OWNER_CHANGED\n")


def test_live_bot_room_owner_auto_start() -> None:
    """When bot is room owner, it should auto-start when minimum players (2) join."""
    client = TcpGameClient(host="127.0.0.1", port=65000)
    bot = LiveBot(client=client, agent=RandomAgent(), bot_name="HostBot", auto_start=True)

    # 1 player in 8-player room: not enough players yet
    status_one = RoomStatus(
        room_id="r-host",
        current_players=1,
        max_players=8,
        player_names=["HostBot"],
    )
    assert bot.handle_event(status_one) is None
    assert bot.state.is_room_owner is True

    # 2nd player joins: since HostBot is room owner, auto-start triggers immediately
    status_two = RoomStatus(
        room_id="r-host",
        current_players=2,
        max_players=8,
        player_names=["HostBot", "GuestPlayer"],
    )
    cmd = bot.handle_event(status_two)
    assert isinstance(cmd, Start)

    # Ownership transferred to another player and then back to bot
    state = LiveGameState(bot_name="BotB")
    state.apply_event(
        RoomStatus(
            room_id="r2",
            current_players=2,
            max_players=4,
            player_names=["Alice", "BotB"],
        )
    )
    assert state.is_room_owner is False
    state.apply_event(RoomOwnerChanged(new_owner_username="BotB"))
    assert state.is_room_owner is True


def test_cli_discovery_and_hosting_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI flags --access-code, --create-room, --public, and --private."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_live_bot.py",
            "--name",
            "HostBot_1",
            "--create-room",
            "PrivateArena",
            "--private",
            "--access-code",
            "SEC4",
        ],
    )
    args = run_live_bot.parse_args()
    assert args.name == "HostBot_1"
    assert args.create_room == "PrivateArena"
    assert args.private is True
    assert args.public is False
    assert args.access_code == "SEC4"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_live_bot.py",
            "--name",
            "PublicHost",
            "--create-room",
            "OpenTable",
            "--public",
        ],
    )
    args_pub = bot_cli.parse_args()
    assert args_pub.create_room == "OpenTable"
    assert args_pub.public is True
    assert args_pub.private is False
