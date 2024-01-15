import pytest

from pydvl.value.games import Game, GlovesGame, MinerGame


@pytest.fixture(scope="module")
def test_game(request) -> Game:
    name, kwargs = request.param
    if name == "miner":
        game = MinerGame(n_players=kwargs["n_players"])
    elif name == "gloves":
        game = GlovesGame(left=kwargs["left"], right=kwargs["right"])
    else:
        raise ValueError(f"Unknown game '{name}'")
    return game
