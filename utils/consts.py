# Output directory names
SAVED_MODEL_DIR = "saved_model/"
PERFORMANCE_DIR = "performance/"
PREDICTIONS_DIR = "predictions/"
EPOCHS_PREDICTIONS_DIR = PREDICTIONS_DIR + "epochs/"
# Model names
RNN = "RNN"
LSTM = "LSTM"
GRU = "GRU"
# Optimizer names
SGD = "SGD"
ADAM = "ADAM"
# Config fields
MODEL_TYPE = "MODEL_TYPE"
OPTIMIZER = "OPTIMIZER"
LEARNING_RATE = "LEARNING_RATE"
DROPOUT = "DROPOUT"
HIDDEN_SIZE = "HIDDEN_SIZE"
NUM_LAYERS = "NUM_LAYERS"
NUM_EPOCHS = "NUM_EPOCHS"
BATCH_SIZE = "BATCH_SIZE"
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
ID_YEAR = "identifier-year"
# Coordinates
LATITUDE = "lat"  # +1 feature
LONGITUDE = "lon"  # +1 feature
# Original time column
TIMESTAMP = "timestamp"
# Derived movement features
# Intra-day features
MEAN_DISTANCE = "daily_mean_distance"
MEAN_VELOCITY = "daily_mean_velocity"
MEAN_BEARING = "daily_mean_bearing"
MEAN_TURN_ANGLE = "daily_mean_turn_angle"
# Inter-day (daily downsampled) features
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
YEAR = "year"
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
    # Intra-day daily averages
    MEAN_DISTANCE,
    MEAN_VELOCITY,
    MEAN_BEARING,
    MEAN_TURN_ANGLE,
    # Inter-day values (day-to-day)
    DISTANCE,
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
# Number of input features: 14
N_FEATURES = len(FEATURE_COLUMNS)
# Setup CSV output columns
CORRECT = "Correct"
PREDICTED = "Predicted"
LABEL = "Label"
OUTPUT_FIELDNAMES = [
    CORRECT,
    PREDICTED,
    LABEL,
    ID_YEAR,
    TIMESTAMP,
] + FEATURE_COLUMNS
