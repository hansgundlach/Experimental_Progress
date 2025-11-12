from experiment_utils import (
    create_multi_seed_experiments,
    create_multi_lr_experiments,
    calculate_transformer_params,
    gen_experim,
    get_base_config,
)


SGD_SCALING = (
    gen_experim(
        32,
        label="32d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=10 ** (-1),
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
    # + gen_experim(
    #     48,
    #     label="48d_sgdbs64",
    #     folder_name="new_sgd_scaling",
    #     learning_rate=1.4e-1,
    #     optimizer="sgd",
    #     sgd_momentum=0.98,
    #     weight_decay=0.0,
    #     token_to_param_ratio=40,
    # )
    + gen_experim(
        64,
        label="64d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=1.77e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
    # setu 96 128 160 192 224 256
    + gen_experim(
        96,
        label="96d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=2.490e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
    + gen_experim(
        128,
        label="128d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=3.16e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
    + gen_experim(
        160,
        label="160d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=3.80e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
    + gen_experim(
        192,
        label="192d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=4.4283e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
    # 224
    + gen_experim(
        224,
        label="224d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=5.03e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
    + gen_experim(
        256,
        label="256d_sgdbs64",
        folder_name="new_sgd_scaling",
        learning_rate=5.623e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        token_to_param_ratio=40,
    )
)

SGD_LR_SCALE = (
    gen_experim(
        32,
        label="32_sgd_nowd",
        folder_name="new_sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
    )
    + gen_experim(
        64,
        label="64_sgd_nowd",
        folder_name="new_sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
    )
    # 128
    + gen_experim(
        128,
        label="128_sgd_nowd",
        folder_name="new_sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
    )
    # + gen_experim(
    #     160,
    #     label="160_sgd",
    #     folder_name="new_sgd_scaling",
    #     learning_rate=10 ** (-1.5),
    #     optimizer="sgd",
    #     sgd_momentum=0.98,
    # )
    + gen_experim(
        256,
        label="256_sgd_nowd",
        folder_name="new_sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
    )
)

SGD_LR_TUNE_FIT = create_multi_lr_experiments(SGD_LR_SCALE, [10**-0.25, 10**-0.5])


HISTORICAL_EXPERIMENTS = (
    gen_experim(
        32,
        label="32_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=5.6e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        weight_decay=0.01,
        pos_encoding="sinusoidal",
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        48,
        label="48_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=2.9e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        64,
        label="64_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=1.86e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        80,
        label="80_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=1.30e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        96,
        label="96_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=9.79e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    # continue scaling experiments 104, 128, 160, 256
    + gen_experim(
        104,
        label="104_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=8.62e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        128,
        label="128_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=6.1996e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        160,
        label="160_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=4.34e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        192,
        label="192_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=3.255e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        224,
        label="224_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=2.5484e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
    + gen_experim(
        256,
        label="256_all_reset",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
        token_to_param_ratio=40,
    )
)

HISTORICAL_LR_STUDY = create_multi_lr_experiments(
    gen_experim(
        160,
        label="160_all_reset",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        256,
        label="256_all_reset",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    ),
    [10**-3.25, 10**-3.5, 10**-3.75],
) + create_multi_lr_experiments(
    gen_experim(
        32,
        label="32_all_reset",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-2.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        weight_decay=0.01,
        pos_encoding="sinusoidal",
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        64,
        label="64_all_reset",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-2.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        96,
        label="96_all_reset",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    ),
    [10**-2.25, 10**-2.5, 10**-2.75, 10**-3, 10**-3.25],
)
# 6+15= 21

# Extrapolated optimal LR for width=160: 2.8386e-03
# Extrapolated optimal LR for width=192: 2.4626e-03
# Extrapolated optimal LR for width=224: 2.1837e-03
# Extrapolated optimal LR for width=256: 1.9679e-03
# # modern variation scaling study
MODERN_SCALING_STUDY = (
    # 32 48 64 80 96 128 160
    gen_experim(
        32,
        label="32_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=9.95e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    # + gen_experim(
    #     48,
    #     label="48_modern",
    #     folder_name="modern_scaling_study",
    #     learning_rate=10**-2,
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     norm_type="rms",
    # )
    + gen_experim(
        64,
        label="64_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=5.79e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    + gen_experim(
        80,
        label="80_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=4.87e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    + gen_experim(
        96,
        label="96_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=4.23e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    # + gen_experim(
    #     104,
    #     label="104_modern",
    #     folder_name="new_modern_scaling_study",
    #     learning_rate=4.23e-3,
    #     modern_bias_0=True,
    #     ff_ratio=2.5,
    #     norm_type="rms",
    #     token_to_param_ratio=40,
    # )
    + gen_experim(
        128,
        label="128_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=3.37e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    + gen_experim(
        160,
        label="160_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=2.83e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    + gen_experim(
        192,
        label="192_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=2.4626e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    + gen_experim(
        224,
        label="224_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=2.1837e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
    + gen_experim(
        256,
        label="256_modern_40",
        folder_name="new_modern_scaling_study",
        learning_rate=1.967e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
    )
)


########################################################
# Kaplan scaling study
########################################################

KAPLAN_SCALING_STUDY = (
    # 32 48 64 80 96 128 160
    gen_experim(
        32,
        label="32_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=9.95e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    # + gen_experim(
    #     48,
    #     label="48_modern",
    #     folder_name="modern_scaling_study",
    #     learning_rate=10**-2,
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     norm_type="rms",
    # )
    + gen_experim(
        64,
        label="64_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=5.79e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    + gen_experim(
        80,
        label="80_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=4.87e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    + gen_experim(
        96,
        label="96_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=4.23e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    # + gen_experim(
    #     104,
    #     label="104_modern",
    #     folder_name="new_modern_scaling_study",
    #     learning_rate=4.23e-3,
    #     modern_bias_0=True,
    #     ff_ratio=2.5,
    #     norm_type="rms",
    #     token_to_param_ratio=40,
    # )
    + gen_experim(
        128,
        label="128_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=3.37e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    + gen_experim(
        160,
        label="160_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=2.83e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    + gen_experim(
        192,
        label="192_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=2.4626e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    + gen_experim(
        224,
        label="224_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=2.1837e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
    + gen_experim(
        256,
        label="256_kaplan_40",
        folder_name="new_modern_scaling_study",
        learning_rate=1.967e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        token_to_param_ratio=40,
        scaling_law="kaplan",
    )
)

# learning rate study
# modern variation scaling study
LR_STUDY_MODERN = (
    # 32 48 64 80 96 128 160
    gen_experim(
        32,
        label="32_modern",
        folder_name="lr_scaling_study",
        learning_rate=10**-2,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        64,
        label="64_modern",
        folder_name="lr_scaling_study",
        learning_rate=10**-2,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        80,
        label="80_modern",
        folder_name="lr_scaling_study",
        learning_rate=10**-2,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        160,
        label="160_modern",
        folder_name="lr_scaling_study",
        learning_rate=10**-2.5,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
)


# # I want to compare streaming vs non-streaming loading for 64d model
# GRAND_EXPERIMENT = (
#     gen_experim(
#         256,
#         label="256_streaming_c4large_40x",
#         learning_rate=1.967e-3,
#         folder_name="debug_streaming_vs_non_streaming",
#         streaming=True,
#         data_path="Datasets/c4_subset_large.txt",
#         modern_bias_0=True,
#         ff_ratio=2.5,
#         token_to_param_ratio=40,
#     )
#     + gen_experim(
#         256,
#         label="256_non_streaming_c4large_40x",
#         learning_rate=1.967e-3,
#         folder_name="debug_streaming_vs_non_streaming",
#         streaming=False,
#         data_path="Datasets/c4_subset_large.txt",
#         modern_bias_0=True,
#         ff_ratio=2.5,
#         token_to_param_ratio=40,
#     )
#     + gen_experim(
#         256,
#         label="256_streaming_c4_40x",
#         learning_rate=1.967e-3,
#         folder_name="debug_streaming_vs_non_streaming",
#         streaming=True,
#         data_path="Datasets/c4_subset.txt",
#         modern_bias_0=True,
#         ff_ratio=2.5,
#         token_to_param_ratio=40,
#     )
#     + gen_experim(
#         256,
#         label="256_non_streaming_c4_40x",
#         learning_rate=1.967e-3,
#         folder_name="debug_streaming_vs_non_streaming",
#         streaming=False,
#         data_path="Datasets/c4_subset.txt",
#         modern_bias_0=True,
#         ff_ratio=2.5,
#         token_to_param_ratio=40,
#     )
# )


# random 32d experiments
# GRAND_EXPERIMENT = gen_experim(
#     32,
#     label="32_random_experiments",
#     learning_rate=1.967e-3,
#     folder_name="debug_streaming_vs_non_streaming",
#     streaming=True,
#     data_path="Datasets/c4_subset_large.txt",
# )


# 64
GRAND_EXPERIMENT = KAPLAN_SCALING_STUDY
