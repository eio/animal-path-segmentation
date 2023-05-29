TERRAIN_TYPE_LOOKUP = {
    "soil_type": {
        # "src": "https://codes.ecmwf.int/grib/param-db/?id=43",
        1: "Coarse",
        2: "Medium",
        3: "Medium fine",
        4: "Fine",
        5: "Very fine",
        6: "Organic",
        7: "Tropical organic",
    },
    "type_of_low_vegetation": {
        # "src": "https://codes.ecmwf.int/grib/param-db/?id=29",
        1: "Crops, Mixed farming",
        2: "Grass",
        7: "Tall grass",
        9: "Tundra",
        10: "Irrigated crops",
        11: "Semidesert",
        13: "Bogs and marshes",
        16: "Evergreen shrubs",
        17: "Deciduous shrubs",
        20: "Water and land mixtures",
        # No land surface vegetation:
        8: "Desert",
        12: "Ice caps and Glaciers",
        14: "Inland water",
        15: "Ocean",
    },
    "type_of_high_vegetation": {
        # "src": "https://codes.ecmwf.int/grib/param-db/?id=30",
        3: "Evergreen needleleaf trees",
        4: "Deciduous needleleaf trees",
        5: "Deciduous broadleaf trees",
        6: "Evergreen broadleaf trees",
        18: "Mixed forest/woodland",
        19: "Interrupted forest",
        # No land surface vegetation:
        8: "Desert",
        12: "Ice caps and Glaciers",
        14: "Inland water",
        15: "Ocean",
    },
}
