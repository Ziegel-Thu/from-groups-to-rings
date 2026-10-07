"""代表性训练与中间层 intervention 的公开配置。"""


ALGEBRAS = {
    "F2_Z7": {
        "p": 2,
        "group_order": 7,
        "factor_names": ["F2", "F8a", "F8b"],
        "factors": [
            [1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
        ],
    },
    "F3_Z4": {
        "p": 3,
        "group_order": 4,
        "factor_names": ["F3a", "F3b", "F9"],
        "factors": [
            [2, 1],
            [1, 1],
            [1, 0, 1],
        ],
    },
    "F2_Z6": {
        "p": 2,
        "group_order": 6,
        "factor_names": ["local-4", "local-16"],
        "factors": [
            [1, 0, 1],
            [1, 0, 1, 0, 1],
        ],
    },
}

D_MODEL = 128
N_HEADS = 4
D_MLP = 512
N_LAYERS = 1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0
BATCH_SIZE = 512
TRAIN_FRACTION = 0.30
TOTAL_STEPS = 100_000
EVAL_INTERVAL = 1_000
GROK_THRESHOLD = 0.95
EARLY_STOP_AFTER_GROK = 10_000

