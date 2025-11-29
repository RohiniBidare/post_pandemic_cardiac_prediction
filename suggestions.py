import json

def load_suggestions():
    with open("suggestions.json", "r") as f:
        return json.load(f)

_sugg = load_suggestions()

def get_suggestions(disease_id, mortality=None):
    disease_id = str(disease_id)

    if disease_id in _sugg:
        data = _sugg[disease_id]
    else:
        data = _sugg["default"]

    # Optional: add urgent advice for high mortality
    if mortality is not None and mortality > 50:
        data["suggestions"] = [
            "High mortality risk detected. Consult emergency services immediately."
        ] + data["suggestions"]

    return data
