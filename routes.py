# routes.py

ROUTE_TO_TRAINS = {
    ("Paris", "Lyon"): ["TGV", "SNCF"],
    ("Lyon", "Paris"): ["TGV", "SNCF"],

    ("Paris", "Marseille"): ["TGV", "SNCF"],
    ("Marseille", "Paris"): ["TGV", "SNCF"],

    ("London", "Paris"): ["Eurostar"],
    ("Paris", "London"): ["Eurostar"],

    ("Paris", "Brussels"): ["Thalys", "Eurostar"],
    ("Brussels", "Paris"): ["Thalys", "Eurostar"],

    ("Brussels", "Amsterdam"): ["Thalys", "Eurostar"],
    ("Amsterdam", "Brussels"): ["Thalys", "Eurostar"],

    ("Berlin", "Munich"): ["ICE", "DB"],
    ("Munich", "Berlin"): ["ICE", "DB"],

    ("Frankfurt", "Cologne"): ["ICE", "DB"],
    ("Cologne", "Frankfurt"): ["ICE", "DB"],

    ("Madrid", "Barcelona"): ["AVE", "Renfe"],
    ("Barcelona", "Madrid"): ["AVE", "Renfe"],

    ("Rome", "Milan"): ["Frecciarossa", "Italo"],
    ("Milan", "Rome"): ["Frecciarossa", "Italo"],

    ("Milan", "Venice"): ["Frecciarossa", "Italo"],
    ("Venice", "Milan"): ["Frecciarossa", "Italo"],

    ("Zurich", "Geneva"): ["SBB"],
    ("Geneva", "Zurich"): ["SBB"]
}


def get_valid_trains(departure: str, arrival: str):
    return ROUTE_TO_TRAINS.get((departure.strip(), arrival.strip()), [])