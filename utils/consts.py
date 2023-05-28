# One-hot encoding for seasonal labels
SEASON_LABELS = {
    "Winter": [1, 0, 0, 0],
    "Spring": [0, 1, 0, 0],
    "Summer": [0, 0, 1, 0],
    "Autumn": [0, 0, 0, 1],
}
# Number of possible labels
N_CATEGORIES = len(SEASON_LABELS)
# Define strings for column/feature names
IDENTIFIER = "individual_id"
# Coordinates
LATITUDE = "lat"  # +1 feature
LONGITUDE = "lon"  # +1 feature
# Original time column
TIMESTAMP = "timestamp"
# Derived movement features
DISTANCE = "dist_from_prev_loc"
VELOCITY = "velocity"
BEARING = "bearing"
TURN_ANGLE = "turn_angle"
# Derived time features
MONTH = "month"  # +1 feature
DAY = "day"  # +1 feature
SINTIME = "sin_time"  # +1 feature
COSTIME = "cos_time"  # +1 feature
# UNIXTIME = "UnixTime"
# YEAR = "Year"  #
SPECIES = "species"
# CONFIDENCE = "confidence"
# # Stopover flag (binary)
# STOPOVER = "stopover"
STATUS = "status"  # the seasonal segmentation label
# Group time features for normalization
TIME_FEATURES = [
    # YEAR,
    MONTH,
    DAY,
    # UNIXTIME,
    SINTIME,
    COSTIME,
]
# Group derived movement features
MOVEMENT_FEATURES = [
    VELOCITY,
    BEARING,
    TURN_ANGLE,
]
# All input feature column names:
FEATURE_COLUMNS = (
    [
        LATITUDE,
        LONGITUDE,
    ]
    + TIME_FEATURES
    + MOVEMENT_FEATURES
)
# Number of input features: 9
N_FEATURES = len(FEATURE_COLUMNS)
