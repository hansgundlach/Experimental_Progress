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

# lstm_layer1_large: parabolic-vertex fit -- eta* = 0.230628 * hidden_dim^(-0.5462), R^2=0.9959
# (standard per Bjorck et al. ICLR 2025 / VaultGemma: quadratic in ln(lr), vertex = eta*)
LSTM_SCALING_LAYER1 = (
    gen_lstm_experim(
        32,
        label="32d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.03473944,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        48,
        label="48d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.02783842,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        64,
        label="64d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.02379058,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        80,
        label="80d",
        folder_name="lstm_layer1",
        learning_rate=0.02106077,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        96,
        label="96d",
        folder_name="lstm_layer1",
        learning_rate=0.01906455,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        128,
        label="128d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01629248,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01442302,
        token_to_param_ratio=40,
    )
    + gen_lstm_experim(
        192,
        label="192d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01305596,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        224,
        label="224d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01200172,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.01115756,
        token_to_param_ratio=40,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        320,
        label="320d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00987730,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        384,
        label="384d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00894110,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        448,
        label="448d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00821913,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        512,
        label="512d",
        folder_name="x1_lstm_layer1",
        learning_rate=0.00764102,
        token_to_param_ratio=40,
        num_layers=1,
        csv_log_interval=200,
    )
)


# lstm_layer2_large: parabolic-vertex fit -- eta* = 0.291135 * hidden_dim^(-0.5886), R^2=0.9745
# (standard per Bjorck et al. ICLR 2025 / VaultGemma: quadratic in ln(lr), vertex = eta*)
#change the validateion interval
LSTM_SCALING_LAYER2 = (
    gen_lstm_experim(
        32,
        label="32d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.03785491,
        token_to_param_ratio=40,
        num_layers=2,

    )
    + gen_lstm_experim(
        48,
        label="48d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.02981743,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="64d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.02517259,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        80,
        label="80d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.02207415,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        96,
        label="96d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01982786,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        128,
        label="128d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01673916,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01467877,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        192,
        label="192d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01318504,
        token_to_param_ratio=40,
        num_layers=2,
    )
    + gen_lstm_experim(
        224,
        label="224d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01204134,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.01113113,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=100,
    )
    + gen_lstm_experim(
        384,
        label="384d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.00876773,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        448,
        label="448d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.00800720,
        token_to_param_ratio=40,
        num_layers=2,
        csv_log_interval=200,
    )
    + gen_lstm_experim(
        512,
        label="512d",
        folder_name="x2_lstm_layer2",
        learning_rate=0.00740193,
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

GRAND_EXPERIMENT = LSTM_SCALING_LAYER2+LSTM_SCALING_LAYER1
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


