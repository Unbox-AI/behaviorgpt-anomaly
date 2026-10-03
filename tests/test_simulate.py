from anomaly.events import Event
from anomaly.simulate import open_cart


def test_open_cart_empties_on_order():
    events = [
        Event("add_to_cart", item="a"),
        Event("add_to_cart", item="b"),
        Event("remove_from_cart", item="b"),
    ]
    assert open_cart(events) == {"a"}
    assert open_cart(events + [Event("order")]) == set()
    assert open_cart(events + [Event("order"), Event("add_to_cart", item="c")]) == {"c"}
