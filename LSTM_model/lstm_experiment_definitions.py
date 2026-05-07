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
# LSTM_SCALING_STUDY_TRADITIONAL_ORIGINAL = (
#     gen_lstm_experim(
#         32,
#         label="32d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=5.6234e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         48,
#         label="48d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=4.0157e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         64,
#         label="64d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=3.1623e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         80,
#         label="80d",
#         folder_name="lstm_layer1",
#         learning_rate=2.627e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         104,
#         label="104d",
#         folder_name="lstm_layer1",
#         learning_rate=2.1130e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         128,
#         label="128d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=1.7783e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         160,
#         label="160d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=1.4775e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         192,
#         label="192d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=1.269e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         224,
#         label="224d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=1.1173e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         256,
#         label="256d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=1.0e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         320,
#         label="320d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=8.3084e-3,
#         token_to_param_ratio=40,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         384,
#         label="384d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=7.141e-3,
#         token_to_param_ratio=40,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         448,
#         label="448d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=6.2829e-3,
#         token_to_param_ratio=40,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         512,
#         label="512d",
#         folder_name="x1_lstm_layer1",
#         learning_rate=5.62e-3,
#         token_to_param_ratio=40,
#         num_layers=1,
#     )
# )


# LSTM_SCALING_LAYER2 = (
#     gen_lstm_experim(
#         32,
#         label="32d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=5.6234e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         48,
#         label="48d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=4.0157e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         64,
#         label="64d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=3.1623e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         80,
#         label="80d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=2.627e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         104,
#         label="104d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=2.1130e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         128,
#         label="128d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=1.7783e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         160,
#         label="160d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=1.4775e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         192,
#         label="192d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=1.269e-2,
#         token_to_param_ratio=40,
#     )
#     + gen_lstm_experim(
#         224,
#         label="224d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=1.1173e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         256,
#         label="256d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=1.0e-2,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         384,
#         label="384d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=7.141e-3,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         448,
#         label="448d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=6.2829e-3,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
#     + gen_lstm_experim(
#         512,
#         label="512d",
#         folder_name="x2_lstm_layer2",
#         learning_rate=5.62e-3,
#         token_to_param_ratio=40,
#         num_layers=2,
#     )
# )

# lstm_fair_layer1_iter9: parabolic+argmin vs non-embedding N
# eta* = 1.145964e+02 * N^(-0.7068), R^2=0.9803
LSTM_SCALING_LAYER1 = (
    gen_lstm_experim(
        32,
        label="32d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.04888174,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        48,
        label="48d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.04359178,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        64,
        label="64d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.03811782,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        80,
        label="80d",
        folder_name="lstm_layer1",
        learning_rate=0.03307955,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        96,
        label="96d",
        folder_name="lstm_layer1",
        learning_rate=0.02870915,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        112,
        label="112d",
        folder_name="lstm_layer1",
        learning_rate=0.02502319,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        128,
        label="128d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.02194855,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01724922,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        192,
        label="192d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01393098,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        224,
        label="224d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01152145,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00972073,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        320,
        label="320d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00725464,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        384,
        label="384d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00567829,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        448,
        label="448d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00460229,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        512,
        label="512d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00383021,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
)


# lstm_fair_layer2_iter9: parabolic+argmin vs non-embedding N
# eta* = 4.899486e+00 * N^(-0.4156), R^2=0.6281
#change the validateion interval
LSTM_SCALING_LAYER2 = (
    gen_lstm_experim(
        32,
        label="32d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.04840321,
        token_to_param_ratio=40,
        num_layers=2,

    )
    + gen_lstm_experim(
        48,
        label="48d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.04329205,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="64d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.03846436,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        80,
        label="80d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.03429395,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        96,
        label="96d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.03080329,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        112,
        label="112d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.02790056,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        128,
        label="128d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.02547667,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.02169970,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        192,
        label="192d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01891657,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        224,
        label="224d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01679012,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01511495,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        384,
        label="384d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01090945,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        448,
        label="448d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.00962043,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        512,
        label="512d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.00862328,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
)


LSTM_LR_STUDY = (
    # gen_lstm_experim(
    #     32,
    #     label="32_lr_study",
    #     folder_name="x1_lstm_lr",
    #     learning_rate=10 ** -(1.5),
    # )
    # + gen_lstm_experim(
    #     64,
    #     label="64_lr_study",
    #     folder_name="x1_lstm_lr",
    #     learning_rate=10 ** -(1.5),
    # )
    # + gen_lstm_experim(
    #     128,
    #     label="128_lr_study",
    #     folder_name="x1_lstm_lr",
    #     learning_rate=10 ** -(1.5),
    # )
    # # 160
    # + gen_lstm_experim(
    #     160,
    #     label="160_lr_study",
    #     folder_name="x1_lstm_lr",
    #     learning_rate=10 ** -(1.5),
    # )
    # + gen_lstm_experim(
    #     256,
    #     label="256_lr_study",
    #     folder_name="x1_lstm_lr",
    #     learning_rate=10 ** -(1.5),
    # )
    gen_lstm_experim(
        320,
        label="320_lr_study",
        folder_name="x1_lstm_layer1",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        384,
        label="384_lr_study",
        folder_name="x1_lstm_layer1",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        448,
        label="448_lr_study",
        folder_name="x1_lstm_layer1",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        512,
        label="512_lr_study",
        folder_name="x1_lstm_layer1",
        learning_rate=10 ** -(1.5),
    )
)


#LSTM 2 layer lr study 
LSTM_2_LAYER_LR_STUDY = (
    gen_lstm_experim(
        256,
        label="256_lr_study",
        folder_name="x2_lstm_lr",
        learning_rate=10 ** -(1.5),
        num_layers=2,
    )
    + gen_lstm_experim(
        384,
        label="384_lr_study",
        folder_name="x2_lstm_lr",
        learning_rate=10 ** -(1.5),
        num_layers=2,
    )
    + gen_lstm_experim(
        448,
        label="448_lr_study",
        folder_name="x2_lstm_lr",
        learning_rate=10 ** -(1.5),
        num_layers=2,
    )
    + gen_lstm_experim(
        512,
        label="512_lr_study",
        folder_name="x2_lstm_lr",
        learning_rate=10 ** -(1.5),
        num_layers=2,
    )
)

# create multi lr experiments for 2 layer lr study
LSTM_2_LAYER_LR_STUDY = create_multi_lr_experiments(
    LSTM_2_LAYER_LR_STUDY,
    [10**-3, 10**-2.75, 10**-2.5, 10**-2.25, 10**-2.0, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1],
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
    input_dropout=0.6,
    hidden_dropout=0.3,
    output_dropout=0.7,
    between_layers_dropout=0.0,
) + gen_lstm_experim(
    64,
    label="64d_standard_no_dropout",
    folder_name="appendix_ablation_study",
    learning_rate=3.1623e-2,
)
NO_DROPOUT_COMPARISON_LR = create_multi_lr_experiments(
    NO_DROPOUT_COMPARISON,
    [10**-2.5, 10**-2.25, 10**-2, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1],
)

########################################################
# LSTM LR TREND EXPERIMENTS
# Layer 1 LRs from lstm_1_layer_all parabolic trend fit:
# log10(eta*) = -0.4223 * log10(N) + 0.6587
########################################################
LSTM_LR_TREND_LAYER1 = (
    gen_lstm_experim(
        16,
        label="16d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.04632708,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        32,
        label="32d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.04419776,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        48,
        label="48d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.04127470,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        64,
        label="64d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.03809509,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        80,
        label="80d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.03500156,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        96,
        label="96d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.03216054,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        112,
        label="112d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.02962589,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        128,
        label="128d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.02739409,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.02372172,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        192,
        label="192d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.02087920,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        224,
        label="224d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01863988,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01684028,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        288,
        label="288d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01536659,
        token_to_param_ratio=40,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        320,
        label="320d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01413933,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        352,
        label="352d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01310213,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        384,
        label="384d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01221422,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        448,
        label="448d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01077347,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        416,
        label="416d",
        folder_name="x1_lstm_layer1_lr_trend",
        learning_rate=0.01144552,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
)

########################################################
# Layer 2 LRs from lstm_2_layer_all parabolic trend fit:
# log10(eta*) = -0.2793 * log10(N) + 0.0312
########################################################
LSTM_LR_TREND_LAYER2 = (
    gen_lstm_experim(
        16,
        label="16d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.05109363,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        32,
        label="32d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.04822875,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        48,
        label="48d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.04474346,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="64d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.04132520,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        80,
        label="80d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.03825730,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        96,
        label="96d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.03559413,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        112,
        label="112d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.03330325,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        128,
        label="128d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.03132972,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.02812647,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        192,
        label="192d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.02564764,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        224,
        label="224d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.02367215,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.02205749,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        288,
        label="288d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.02070967,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        320,
        label="320d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.01956473,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        352,
        label="352d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.01857779,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        384,
        label="384d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.01771646,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        448,
        label="448d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.01628066,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        416,
        label="416d",
        folder_name="x2_lstm_layer2_lr_trend",
        learning_rate=0.01695679,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
)

GRAND_EXPERIMENT = LSTM_LR_TREND_LAYER2 + LSTM_LR_TREND_LAYER1
# =================================================== LR SWEEPS ===================================================

# GRAND_EXPERIMENT = NO_DROPOUT_COMPARISON_LR
# GRAND_EXPERIMENT = (
#     gen_lstm_experim(
#         320,
#         label="320d_grad_clipping_0.25",
#         folder_name="debug_runs",
#         learning_rate=0.00994133,
#         token_to_param_ratio=40,
#         gradient_clipping=0.25,
#         num_layers=1,

#     )
#     + gen_lstm_experim(
#         320,
#         label="320d_grad_clipping_1",
#         folder_name="debug_runs",
#         learning_rate=0.00994133,
#         token_to_param_ratio=40,
#         gradient_clipping=1,
#         num_layers=1,
#     )
# )


# 48 experimetns

