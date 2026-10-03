import random
from collections import Counter

from anomaly.api import Ranked
from anomaly.suggest import near_duplicate, sample_next, title_tokens


def ranked(names: list[str], step: float = 0.01) -> list[Ranked]:
    return [
        Ranked(f"i{n}", 1.5 - n * step, {"name": name}) for n, name in enumerate(names)
    ]


def test_variants_are_near_duplicates_and_other_products_are_not():
    a = title_tokens(
        "D'Addario Guitar Strings - XL Nickel Electric Guitar Strings EXL110"
    )
    b = title_tokens(
        "D'Addario Guitar Strings - XL Nickel Electric Guitar Strings EXL115"
    )
    c = title_tokens("Fender Guitar Picks, 12 Pack, Medium")
    assert near_duplicate(a, b)
    assert not near_duplicate(a, c)


def test_skips_variants_of_touched_items():
    items = ranked(
        [
            "Ernie Ball Regular Slinky Nickel Wound Electric Guitar Strings 3 Pack",
            "Ernie Ball Regular Slinky Nickel Wound Electric Guitar Strings 6 Pack",
            "Snark Clip-On Chromatic Guitar Tuner",
        ]
    )
    pick = sample_next(
        items,
        ["Ernie Ball Regular Slinky Nickel Wound Electric Guitar Strings 10 Pack"],
        tau=0.01,
        temperature=1.0,
        rng=random.Random(0),
    )
    assert pick.data["name"].startswith("Snark")


def test_temperature_spreads_the_picks():
    words = [
        "lamp",
        "kettle",
        "drill",
        "tent",
        "novel",
        "router",
        "blender",
        "mat",
        "helmet",
        "scarf",
        "camera",
        "puzzle",
        "speaker",
        "skillet",
        "hose",
    ]
    names = [f"{w} {n}" for n, w in enumerate(words)]
    items = ranked(names, step=0.004)

    def spread(temperature: float) -> int:
        rng = random.Random(1)
        picks = Counter(
            sample_next(items, [], tau=0.0064, temperature=temperature, rng=rng).id
            for _ in range(300)
        )
        return len(picks)

    assert spread(1.0) < spread(4.0)


def test_repetition_penalty_moves_past_more_of_the_same():
    items = ranked(
        [
            "Elixir Strings Nickel Plated Steel Electric Guitar Strings",
            "Martin Authentic Acoustic Guitar Strings Extra Light",
            "Jim Dunlop Standard Yellow Guitar Pick 12 Pack",
        ],
        step=0.005,
    )
    touched = ["D'Addario Guitar Strings XL Nickel Electric Guitar Strings"]
    rng = random.Random(0)
    picks = Counter(
        sample_next(items, touched, tau=0.0064, temperature=3, rng=rng).id
        for _ in range(300)
    )
    assert picks.most_common(1)[0][0] == "i2"
