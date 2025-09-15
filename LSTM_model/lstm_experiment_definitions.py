# lstm_experiment_definitions.py
import math
import copy
from lstm_experiment_utils import (
    gen_lstm_experim,
    create_multi_seed_lstm_experiments,
    create_multi_lr_lstm_experiments,
    create_multi_lr_experiments,
    gen_lstm_experim,
    calculate_lstm_params,
    get_lstm_base_config,
)


TEST_EXPERIMENT = gen_lstm_experim(
    16, label="lstm_16d_test_experiment", learning_rate=0.01
)


NARROW_LR_SWEEP = [
    10 ** (-3),
    10 ** (-2.5),
    10 ** (-2),
    10 ** (-1.5),
    1e-1,
]


LSTM_LR_TUNE_MUP_STANDARD = create_multi_lr_experiments(
    gen_lstm_experim(
        32, label="32d_lstm_mup", learning_rate=0.01, use_mup=True, mup_base_width=32
    ),
    NARROW_LR_SWEEP,
)


LSTM_SCALING_EXPERIMENTS_OPTIMAL_LR = EXPERIMENTS = (
    gen_lstm_experim(
        32, label="32d_lstm_experiment", folder_name="lstm_scaling", learning_rate=0.01
    )
    # + gen_lstm_experim(
    #     48, label="48d_lstm_experiment", folder_name="lstm_scaling", learning_rate=0.01
    # )
    # + gen_lstm_experim(
    #     64, label="64d_lstm_experiment", folder_name="lstm_scaling", learning_rate=0.01
    # )
)

LSTM_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR = EXPERIMENTS = (
    gen_lstm_experim(
        32,
        label="32d_lstm_sgd_experiment",
        folder_name="lstm_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_lstm_experim(
        48,
        label="48d_lstm_sgd_experiment",
        folder_name="lstm_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_lstm_experim(
        64,
        label="64d_lstm_sgd_experiment",
        folder_name="lstm_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
)


LSTM_ALL_SCALE_LR_TUNE = create_multi_lr_experiments(
    LSTM_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
)


LSTM_SGD_ALL_SCALE_LR_TUNE = create_multi_lr_experiments(
    LSTM_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
)


# TBPTT comparison experiments
TBPTT_COMPARISON_EXPERIMENTS = (
    gen_lstm_experim(
        32,
        label="32d_lstm_tbptt_default",
        folder_name="tbptt_comparison",
        learning_rate=0.01,
        use_tbptt=True,
        tbptt_length=128,
        tbptt_stride=128,
    )
    + gen_lstm_experim(
        32,
        label="32d_lstm_tbptt_disabled",
        folder_name="tbptt_comparison",
        learning_rate=0.01,
        use_tbptt=False,
    )
    + gen_lstm_experim(
        32,
        label="32d_lstm_tbptt_64_length",
        folder_name="tbptt_comparison",
        learning_rate=0.01,
        use_tbptt=True,
        tbptt_length=64,
        tbptt_stride=64,
    )
)

# Diagnostic experiments to fix scaling plateau
LSTM_SCALING_DIAGNOSTIC = (
    # Test higher learning rates
    gen_lstm_experim(
        32,
        label="32d_lstm_lr_aggressive",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,  # 5x higher than current
        warmup_frac=0.05,  # 5x longer warmup
        use_tbptt=False,  # Disable TBPTT initially
    )
    + gen_lstm_experim(
        32,
        label="32d_lstm_lr_very_aggressive",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.03,  # 15x higher
        warmup_frac=0.1,  # 10x longer warmup
        use_tbptt=False,
    )
    # Test lower dropout
    + gen_lstm_experim(
        32,
        label="32d_lstm_low_dropout",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.05,  # Much lower
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.1,  # Lower output dropout
        warmup_frac=0.05,
        use_tbptt=False,
    )
    # Test longer sequences with TBPTT
    + gen_lstm_experim(
        32,
        label="32d_lstm_long_seq",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        sequence_length=256,  # 2x longer sequences
        tbptt_length=128,  # Half the sequence length
        warmup_frac=0.05,
        use_tbptt=True,
    )
    #
)


# test no dropu
GRAND_EXPERIMENT = (
    gen_lstm_experim(
        32,
        label="32dyy_lstm_no_dropout_002_warmup",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=False,
    )
    # Uncomment to add low dropout experiment
    # + gen_lstm_experim(
    #     32,
    #     label="32d_lstm_low_dropout_0.02_warmup",
    #     folder_name="lstm_scaling_diagnostic",
    #     learning_rate=0.01,
    #     input_dropout=0.05,   # Low input dropout
    #     hidden_dropout=0.0,   # No hidden dropout
    #     output_dropout=0.1,   # Low output dropout
    #     warmup_frac=0.02,
    #     use_tbptt=False,
    # )
    + gen_lstm_experim(
        32,
        label="32dyy_lstm_no_dropout_002_warmup_testtbptt",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
    )
)

# lr_tune_experiments standard
# LSTM_LR_TUNE_STANDARD = create_multi_lr_experiments(
#     gen_lstm_experim(
#         32, label="32d_standard", learning_rate=0.01, use_mup=True, mup_base_width=32
#     ),
#     NARROW_LR_SWEEP,
# )


# ========= Experiment definitions (customize labels & overrides below) =========
# TEST_EXPERIMENTS = [
#     {
#         "name": "lstm_september_testing",
#         "subexperiments": [
#             {
#                 "label": "lstm_with_tokens_estimate",
#                 "overrides": {"learning_rate": 0.001 * math.sqrt(4), "hidden_size": 16},
#             },
#         ],
#     },
# ]


# LSTM_OPTIMAL_SCALING = [
#     {
#         "name": "lstm_optimal_scaling",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                 },
#             },
#             {
#                 "label": "lstm_24d",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193e6 * 2,
#                     "hidden_size": 24,
#                 },
#             },
#             {
#                 "label": "lstm_32d",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 258e6 * 2,
#                     "hidden_size": 32,
#                 },
#             },
#             {
#                 "label": "lstm_48d",
#                 "overrides": {
#                     "learning_rate": 1e-2,
#                     "max_characters": 388e6 * 2,
#                     "hidden_size": 48,
#                 },
#             },
#             {
#                 "label": "lstm_64d",
#                 "overrides": {
#                     "learning_rate": 1e-2,
#                     "max_characters": 519e6 * 2,
#                     "hidden_size": 64,
#                 },
#             },
#         ],
#     },
# ]


# LSTM_SGD_OPTIMAL_SCALING = [
#     {
#         "name": "lstm_sgd_optimal_scaling",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_24d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 193.7e6 * 2,
#                     "hidden_size": 24,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_32d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 258.6e6 * 2,
#                     "hidden_size": 32,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_48d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 388.8e6 * 2,
#                     "hidden_size": 48,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_64d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 519.9e6 * 2,
#                     "hidden_size": 64,
#                     "optimizer": "sgd",
#                 },
#             },
#         ],
#     },
# ]

# LSTM_SGD_MUP_SCALING = [
#     {
#         "name": "lsmt_sgd_mup_scaling",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_24d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 193.7e6 * 2,
#                     "hidden_size": 24,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_32d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 258.6e6 * 2,
#                     "hidden_size": 32,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_48d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 388.8e6 * 2,
#                     "hidden_size": 48,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_64d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 519.9e6 * 2,
#                     "hidden_size": 64,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#         ],
#     },
# ]

# LSTM_VARIATIONS = [
#     {
#         "name": "lstm_variations",
#         "subexperiments": [
#             {
#                 "label": "lstm_24d_layernorm",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                 },
#             },
#             {
#                 "label": "lstm_24d_3_layers",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "use_layer_norm": True,
#                     "num_layers": 3,
#                 },
#             },
#             {
#                 "label": "lstm_24d_no_layer_norm",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "use_layer_norm": False,
#                 },
#             },
#             {
#                 "label": "lstm_24d_cosine_warmup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "lr_schedule": "cosine_warmup",
#                 },
#             },
#             {
#                 "label": "lstm_24d_inverse_sqrt",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "lr_schedule": "inverse_sqrt",
#                 },
#             },
#         ],
#     },
# ]


# LSTM_MUP_SCALING_EXPERIMENTS = [
#     {
#         "name": "muP_scaling_experiments",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                     "seed": 123,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Base width for muP scaling
#                 },
#             },
#             {
#                 "label": "lstm_24d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "hidden_size": 24,
#                     "max_characters": 193.7e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Base width for muP scaling
#                 },
#             },
#             {
#                 "label": "lstm_32d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),  # Same base LR as 16d
#                     "hidden_size": 32,
#                     "max_characters": 258.6e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Same base width - muP should handle scaling
#                 },
#             },
#             {
#                 "label": "lstm_48d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),  # Same base LR as 16d
#                     "hidden_size": 48,
#                     "max_characters": 388.8e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Same base width - muP should handle scaling
#                 },
#             },
#             {
#                 "label": "lstm_64d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),  # Same base LR - muP handles scaling
#                     "hidden_size": 64,
#                     "max_characters": 519.9e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Same base width
#                 },
#             },
#         ],
#     },
# ]
