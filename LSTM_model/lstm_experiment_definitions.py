# lstm_experiment_definitions.py
import math
import copy
from lstm_experiment_utils import (
    gen_lstm_experim,
    create_multi_seed_lstm_experiments,
    create_multi_lr_experiments,
    gen_lstm_experim,
    calculate_lstm_params,
    get_lstm_base_config,
)

# =================================================== FUNDAMENTAL EXPERIMENTS ===================================================
LSTM_SCALING_STUDY_TRADITIONAL = (
    gen_lstm_experim(
        32,
        label="32d",
        folder_name="lstm_layer1",
        learning_rate=5.6234e-2,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        48,
        label="48d",
        folder_name="lstm_layer1",
        learning_rate=4.0157e-2,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        64,
        label="64d_huge_data_160",
        folder_name="x1_lstm_layer1",
        learning_rate=3.1623e-2,
        token_to_param_ratio=40,
    )
    # + gen_lstm_experim(
    #     80,
    #     label="80d",
    #     folder_name="lstm_layer1",
    #     learning_rate=2.627e-2,
    #     token_to_param_ratio=40,
    # )
    # + gen_lstm_experim(
    #     104,
    #     label="104d",
    #     folder_name="lstm_layer1",
    #     learning_rate=2.1130e-2,
    #     token_to_param_ratio=40,
    # )
    + gen_lstm_experim(
        128,
        label="128d_80_huge_data",
        folder_name="x1_lstm_layer1",
        learning_rate=1.7783e-2,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="lstm_layer1",
        learning_rate=1.4775e-2,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        192,
        label="192d",
        folder_name="lstm_layer1",
        learning_rate=1.269e-2,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        224,
        label="224d",
        folder_name="lstm_layer1",
        learning_rate=1.1173e-2,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="lstm_layer1",
        learning_rate=1.0e-2,
        token_to_param_ratio=40,
    )
)


LSTM_LR_STUDY = (
    gen_lstm_experim(
        32,
        label="32_lr_study",
        folder_name="x1_lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        64,
        label="64_lr_study",
        folder_name="x1_lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        128,
        label="128_lr_study",
        folder_name="x1_lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    # 160
    + gen_lstm_experim(
        160,
        label="160_lr_study",
        folder_name="x1_lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        256,
        label="256_lr_study",
        folder_name="x1_lstm_lr",
        learning_rate=10 ** -(1.5),
    )
)

# total experiments 5*7=35
LSTM_LR_STUDY = create_multi_lr_experiments(
    LSTM_LR_STUDY, [10**-2.5, 10**-2.25, 10**-2, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1]
)


# MELSIS dropout vs no dropout
NO_DROPOUT_COMPARISON = gen_lstm_experim(
    64,
    label="64d_melis_dropout",
    folder_name="appendix_ablation_study",
    learning_rate=3.1623e-2,
    token_to_param_ratio=40,
    input_dropout=0.6,
    hidden_dropout=0.3,
    output_dropout=0.7,
    between_layers_dropout=0.0,
) + gen_lstm_experim(
    64,
    label="64d_standard_no_dropout",
    folder_name="appendix_ablation_study",
    learning_rate=3.1623e-2,
    token_to_param_ratio=40,
)
NO_DROPOUT_COMPARISON_LR = create_multi_lr_experiments(
    NO_DROPOUT_COMPARISON,
    [10**-2.5, 10**-2.25, 10**-2, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1],
)


# =================================================== LR SWEEPS ===================================================

GRAND_EXPERIMENT = LSTM_LR_STUDY

# 48 experimetns
