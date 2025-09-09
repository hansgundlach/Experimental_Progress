# lstm_experiment_definitions.py
import math
import copy

# ========= Experiment definitions (customize labels & overrides below) =========
TEST_EXPERIMENTS = [
    {
        "name": "LSTM_benchmark",
        "subexperiments": [
            {
                "label": "LSTM_1.6M_Benchmark",
                "overrides": {"learning_rate": 0.001 * math.sqrt(4), "hidden_size": 16},
            },
        ],
    },
]


LSTM_OPTIMAL_SCALING = [
    {
        "name": "lstm_optimal_scaling",
        "subexperiments": [
            {
                "label": "lstm_16d",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "hidden_size": 16,
                    "max_characters": 129e6 * 2,
                },
            },
            {
                "label": "lstm_24d",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193e6 * 2,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_32d",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 258e6 * 2,
                    "hidden_size": 32,
                },
            },
            {
                "label": "lstm_48d",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 388e6 * 2,
                    "hidden_size": 48,
                },
            },
            {
                "label": "lstm_64d",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 519e6 * 2,
                    "hidden_size": 64,
                },
            },
        ],
    },
]


LSTM_SGD_OPTIMAL_SCALING = [
    {
        "name": "lstm_sgd_optimal_scaling",
        "subexperiments": [
            {
                "label": "lstm_16d_sgd",
                "overrides": {
                    "learning_rate": 1e-1,
                    "hidden_size": 16,
                    "max_characters": 129e6 * 2,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "lstm_24d_sgd",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 193.7e6 * 2,
                    "hidden_size": 24,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "lstm_32d_sgd",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 258.6e6 * 2,
                    "hidden_size": 32,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "lstm_48d_sgd",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 388.8e6 * 2,
                    "hidden_size": 48,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "lstm_64d_sgd",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 519.9e6 * 2,
                    "hidden_size": 64,
                    "optimizer": "sgd",
                },
            },
        ],
    },
]

LSTM_SGD_MUP_SCALING = [
    {
        "name": "lsmt_sgd_mup_scaling",
        "subexperiments": [
            {
                "label": "lstm_16d_sgd_mup",
                "overrides": {
                    "learning_rate": 1e-1,
                    "hidden_size": 16,
                    "max_characters": 129e6 * 2,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "lstm_24d_sgd_mup",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 193.7e6 * 2,
                    "hidden_size": 24,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "lstm_32d_sgd_mup",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 258.6e6 * 2,
                    "hidden_size": 32,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "lstm_48d_sgd_mup",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 388.8e6 * 2,
                    "hidden_size": 48,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "lstm_64d_sgd_mup",
                "overrides": {
                    "learning_rate": 1e-1,
                    "max_characters": 519.9e6 * 2,
                    "hidden_size": 64,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
        ],
    },
]

LSTM_VARIATIONS = [
    {
        "name": "lstm_variations",
        "subexperiments": [
            {
                "label": "lstm_24d_layernorm",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_24d_3_layers",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                    "num_layers": 3,
                },
            },
            {
                "label": "lstm_24d_no_layer_norm",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": False,
                },
            },
            {
                "label": "lstm_24d_cosine_warmup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "lr_schedule": "cosine_warmup",
                },
            },
            {
                "label": "lstm_24d_inverse_sqrt",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "lr_schedule": "inverse_sqrt",
                },
            },
        ],
    },
]


LSTM_MUP_SCALING_EXPERIMENTS = [
    {
        "name": "muP_scaling_experiments",
        "subexperiments": [
            {
                "label": "lstm_16d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "hidden_size": 16,
                    "max_characters": 129e6 * 2,
                    "seed": 123,
                    "use_mup": True,
                    "mup_base_width": 16,  # Base width for muP scaling
                },
            },
            {
                "label": "lstm_24d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "hidden_size": 24,
                    "max_characters": 193.7e6 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,  # Base width for muP scaling
                },
            },
            {
                "label": "lstm_32d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),  # Same base LR as 16d
                    "hidden_size": 32,
                    "max_characters": 258.6e6 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,  # Same base width - muP should handle scaling
                },
            },
            {
                "label": "lstm_48d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),  # Same base LR as 16d
                    "hidden_size": 48,
                    "max_characters": 388.8e6 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,  # Same base width - muP should handle scaling
                },
            },
            {
                "label": "lstm_64d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),  # Same base LR - muP handles scaling
                    "hidden_size": 64,
                    "max_characters": 519.9e6 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,  # Same base width
                },
            },
        ],
    },
]
