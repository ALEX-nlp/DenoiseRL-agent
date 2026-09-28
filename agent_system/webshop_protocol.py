"""WebShop data profiles; independent of simulator and training dependencies."""

WEBSHOP_DATA_PROFILES = {
    "gigpo_small": {
        "products": "items_shuffle_1000.json",
        "attributes": "items_ins_v2_1000.json",
        "human_goals": False,
        "index": "indexes_gigpo_small",
        "train_start": 500,
    },
    "full_human": {
        "products": "items_shuffle.json",
        "attributes": "items_ins_v2.json",
        "human_goals": True,
        "index": "indexes",
        "train_start": 1500,
    },
}


def webshop_data_profile(name):
    try:
        return WEBSHOP_DATA_PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown WebShop data profile: {name!r}") from None
