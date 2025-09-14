from experiment_utils import (
    create_multi_seed_experiments,
    create_multi_lr_experiments,
    calculate_transformer_params,
    gen_experim,
    get_base_config,
)

NARROW_LR_SWEEP = [
    10 ** (-3),
    10 ** (-2.5),
    10 ** (-2),
    10 ** (-1.5),
    10 ** (-1),
]

TEST_EXPERIMENT = gen_experim(
    16,
    label="junkjunk_16d_speed_experiment_new_token_limit",
    learning_rate=0.01,
    token_limit=8056480,
)

TRANSFORMER_LR_TUNE_MUP_STANDARD = create_multi_lr_experiments(
    gen_experim(
        32,
        label="32d_mup",
        folder_name="zeta_mup_base_tune",
        learning_rate=0.01,
        use_mup=True,
        mup_base_width=32,
    ),
    NARROW_LR_SWEEP,
)

TRANSFORMER_LR_TUNE_MUP_SGD = create_multi_lr_experiments(
    gen_experim(
        32,
        label="32d_mup_sgd",
        folder_name="zeta_mup_base_tune",
        learning_rate=0.01,
        use_mup=True,
        mup_base_width=32,
        optimizer="sgd",
    ),
    NARROW_LR_SWEEP,
)

TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR = (
    gen_experim(
        32,
        label="vanilla_32d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=1e-1,
    )
    + gen_experim(
        40,
        label="vanilla_40d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        48,
        label="vanilla_48d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        56,
        label="vanilla_56d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        64,
        label="vanilla_64d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        80,
        label="vanilla_80d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-1),
    )
)

# ROTARY EXPERIMENTS


TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR_NO_ROTARY = (
    gen_experim(
        32,
        label="vanilla_32d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=1e-2,
        pos_encoding="sinusoidal",
    )
    # + gen_experim(
    #     40,
    #     label="vanilla_40d_no_rot",
    #     folder_name="vanilla_scaling_no_rotary",
    #     learning_rate=10 ** (-2),
    #     pos_encoding="sinusoidal",
    # )
    + gen_experim(
        48,
        label="vanilla_48d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        56,
        label="vanilla_56d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        64,
        label="vanilla_64d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
)


TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP = (
    gen_experim(
        32,
        label="vanilla_32d_rmsprop",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-1.5),
        optimizer="rmsprop",
    )
    + gen_experim(
        48,
        label="vanilla_48d_rmsprop",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-2),
        optimizer="rmsprop",
    )
    + gen_experim(
        56,
        label="vanilla_56d_rmsprop",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-2),
        optimizer="rmsprop",
    )
    + gen_experim(
        64,
        label="vanilla_64d_rmsprop",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-2),
        optimizer="rmsprop",
    )
)


TRANSFORMER_SCALING_EXPERIMENTS_OTHER_SCALES = (
    gen_experim(
        16,
        label="vanilla_16d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        24,
        label="vanilla_24d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-2),
    )
    + gen_experim(
        72,
        label="vanilla_72d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        80,
        label="vanilla_80d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        96,
        label="vanilla_96d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-3),
    )
    + gen_experim(
        128,
        label="vanilla_128d",
        folder_name="vanilla_scaling_optimal_lr",
        learning_rate=10 ** (-3),
    )
)


TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR = (
    # gen_experim(
    #     32,
    #     label="optimal_lr_sgd_32d",
    #     folder_name="optimal_lr_sgd_scaling",
    #     learning_rate=10 ** (-1),
    #     optimizer="sgd",
    # )
    # + gen_experim(
    #     48,
    #     label="optimal_lr_sgd_48d",
    #     folder_name="optimal_lr_sgd_scaling",
    #     learning_rate=0.1,
    #     optimizer="sgd",
    # )
    # + gen_experim(
    #     56,
    #     label="optimal_lr_sgd_56d",
    #     folder_name="optimal_lr_sgd_scaling",
    #     learning_rate=10 ** (-1),
    #     optimizer="sgd",
    # )
    # + gen_experim(
    #     64,
    #     label="optimal_lr_sgd_64d",
    #     folder_name="optimal_lr_sgd_scaling",
    #     learning_rate=10 ** (-1),
    #     optimizer="sgd",
    # )
    # scale the experiments from 72 to 128d
    gen_experim(
        72,
        label="optimal_lr_sgd_72d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
    )
    + gen_experim(
        80,
        label="optimal_lr_sgd_80d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=10 ** (-1),
        optimizer="sgd",
    )
    + gen_experim(
        96,
        label="optimal_lr_sgd_96d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
    )
    + gen_experim(
        128,
        label="optimal_lr_sgd_128d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=10 ** (-1),
        optimizer="sgd",
    )
)


# TRANSFORMER_SCALING_EXPERIMENTS_MUP = (
#     gen_experim(
#         32,
#         label="mup_32d",
#         folder_name="mup_scaling_experiments",
#         learning_rate=10 ** (-1),
#         use_mup=True,
#         mup_base_width=32,
#     )
#     + gen_experim(
#         40,
#         label="mup_40d",
#         folder_name="mup_scaling_experiments",
#         learning_rate=10 ** (-1),
#         use_mup=True,
#         mup_base_width=32,
#     )
#     + gen_experim(
#         48,
#         label="mup_48d",
#         folder_name="mup_scaling_experiments",
#         learning_rate=10 ** (-1),
#         use_mup=True,
#         mup_base_width=32,
#     )
#     + gen_experim(
#         56,
#         label="mup_56d",
#         folder_name="mup_scaling_experiments",
#         learning_rate=10 ** (-1),
#         use_mup=True,
#         mup_base_width=32,
#     )
#     + gen_experim(
#         64,
#         label="mup_64d",
#         folder_name="mup_scaling_experiments",
#         learning_rate=10 ** (-1),
#         use_mup=True,
#         mup_base_width=32,
#     )
# )


TRANSFORMER_ALL_SCALE_LR_TUNE = create_multi_lr_experiments(
    TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
)
TRANSFORMER_SGD_ALL_SCALE_LR_TUNE = create_multi_lr_experiments(
    TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
)


# activation variation experiments
ACTIVATION_EXPERIMENTS = (
    gen_experim(
        16, label="16d_activation_experiment", learning_rate=0.01, activation="gelu"
    )
    + gen_experim(
        16, label="16d_activation_experiment", learning_rate=0.01, activation="relu"
    )
    + gen_experim(
        16, label="16d_activation_experiment", learning_rate=0.01, activation="swiglu"
    )
)

# optimizer variation experiments ?
# positional encoding experiments
# norm experiments
# lr_schedule expeiremtns
# initalizaiton experiments
INITIALIZATION_EXPERIMENTS = (
    gen_experim(
        16,
        label="16d_initialization_experiment",
        learning_rate=0.01,
        init_scheme="xavier_normal",
    )
    + gen_experim(
        16,
        label="16d_initialization_experiment",
        learning_rate=0.01,
        init_scheme="kaiming_normal",
    )
    + gen_experim(
        16,
        label="16d_initialization_experiment",
        learning_rate=0.01,
        init_scheme="transformer",
    )
)
# experiment without rotary + sgd


# lr_tune_experiments standard
TRANSFORMER_LR_TUNE_STANDARD = create_multi_lr_experiments(
    gen_experim(
        32, label="32d_standard", learning_rate=0.01, use_mup=True, mup_base_width=32
    ),
    NARROW_LR_SWEEP,
)


# TRANSFORMER_LR_TUNE_MUP_STANDARD = create_multi_lr_experiments(
#     gen_experim(
#         32, label="32d_mup", learning_rate=0.01, use_mup=True, mup_base_width=32
#     ),
#     NARROW_LR_SWEEP,
# )
# scaling_experiments = (
#     TRANSFORMER_SCALING_EXPERIMENTS_MUP
#     + TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR
#     + TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR
# )


# GRAND_EXPERIMENT = (
#     TRANSFORMER_LR_TUNE_MUP_STANDARD
#     + TRANSFORMER_LR_TUNE_MUP_SGD
#     + TRANSFORMER_SGD_ALL_SCALE_LR_TUNE
# )


# GRAND_EXPERIMENT = TRANSFORMER_SCALING_EXPERIMENTS_MUP+TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR+TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR

# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     TRANSFORMER_SCALING_EXPERIMENTS_OTHER_SCALES, NARROW_LR_SWEEP
# )

# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(
#         TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
#     )
#     + create_multi_lr_experiments(
#         TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP, NARROW_LR_SWEEP
#     )
#     + create_multi_lr_experiments(
#         TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
#     )
# )


# Grand lr tune experiments
# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(
#         TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP, NARROW_LR_SWEEP
#     )
#     + create_multi_lr_experiments(
#         TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
#     )
#     + create_multi_lr_experiments(
#         TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
#     )
# )
# 6*5+6


# GRAND_EXPERIMENT = (
#     TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP
#     + TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR
#     + TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR
#     + TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR_NO_ROTARY
# )


VARIATION_EXPERIMENTS = (
    #     gen_experim(
    #         32,
    #         label="lower min lr",
    #         folder_name="variation_experiments_no_rot",
    #         learning_rate=1e-2,
    #         pos_encoding="sinusoidal",
    #         min_lr_multiplier=0.000001,
    #     )
    #     + gen_experim(
    #         32,
    #         label="cosine annealing simple",
    #         folder_name="variation_experiments_no_rot",
    #         learning_rate=1e-2,
    #         pos_encoding="sinusoidal",
    #         lr_schedule="cosine",
    #     )
    #     + gen_experim(
    #         32,
    #         label="longer warmup",
    #         folder_name="variation_experiments_no_rot",
    #         learning_rate=1e-2,
    #         pos_encoding="sinusoidal",
    #         warmup_frac=0.1,
    #     )
    #     + gen_experim(
    #         32,
    #         label="standard sin",
    #         folder_name="variation_experiments_no_rot",
    #         learning_rate=1e-2,
    #         pos_encoding="sinusoidal",
    #     )
    # gen_experim(
    #     32,
    #     label="2 heads",
    #     folder_name="variation_experiments_no_rot",
    #     learning_rate=1e-2,
    #     num_heads=2,
    #     pos_encoding="sinusoidal",
    # )
    # + gen_experim(
    #     32,
    #     label="4 heads",
    #     folder_name="variation_experiments_no_rot",
    #     learning_rate=1e-2,
    #     num_heads=4,
    #     pos_encoding="sinusoidal",
    #     )
    #
    # gen_experim(
    #     32,
    #     label="sgd smaller",
    #     folder_name="variation_experiments",
    #     learning_rate=1e-2,
    #     num_heads=2,
    #     optimizer="sgd",
    # )
    # + gen_experim(
    #     32,
    #     label="sgd 4 head",
    #     folder_name="variation_experiments",
    #     learning_rate=1e-1,
    #     num_heads=4,
    #     optimizer="sgd",
    # )
    # + gen_experim(
    #     32,
    #     label="sgd_cosine_annealing",
    #     folder_name="variation_experiments",
    #     learning_rate=1e-1,
    #     num_heads=2,
    #     optimizer="sgd",
    #     lr_schedule="cosine",
    # )
    # + gen_experim(
    #     32,
    #     label="sgd_cosine_annealing_small_lr",
    #     folder_name="variation_experiments",
    #     learning_rate=1e-2,
    #     num_heads=2,
    #     optimizer="sgd",
    #     lr_schedule="cosine",
    # )
    # + gen_experim(
    #     32,
    #     label="long warmup",
    #     folder_name="variation_experiments",
    #     learning_rate=1e-1,
    #     num_heads=2,
    #     optimizer="sgd",
    #     warmup_frac=0.1,
    # )
    gen_experim(
        32,
        label="sgd smaller 0.99 momentum",
        folder_name="variation_experiments",
        learning_rate=1e-2,
        num_heads=2,
        optimizer="sgd",
        sgd_momentum=0.99,
    )
    + gen_experim(
        32,
        label="sgd 0.99 momentum",
        folder_name="variation_experiments",
        learning_rate=1e-1,
        num_heads=2,
        optimizer="sgd",
        sgd_momentum=0.99,
    )
    + gen_experim(
        32,
        label="sgd nesterov ",
        folder_name="variation_experiments",
        learning_rate=1e-1,
        num_heads=2,
        optimizer="sgd",
        sgd_nesterov=True,
    )
    + gen_experim(
        32,
        label="sgd nesterov 0.95 momentum",
        folder_name="variation_experiments",
        learning_rate=1e-1,
        num_heads=2,
        optimizer="sgd",
        sgd_nesterov=True,
        momentum=0.95,
    )
    + gen_experim(
        32,
        label="sgd no momentum",
        folder_name="variation_experiments",
        learning_rate=1e-1,
        num_heads=2,
        optimizer="sgd",
        sgd_nesterov=True,
        momentum=0,
    )
    + gen_experim(
        32,
        label="sgd no momentum lr 1e-2",
        folder_name="variation_experiments",
        learning_rate=1e-2,
        num_heads=2,
        optimizer="sgd",
        sgd_nesterov=True,
        momentum=0,
    )
    # + gen_experim(
    #     32,
    #     label="sgd_cosine_annealing_small_lr",
    #     folder_name="variation_experiments",
    #     learning_rate=1e-2,
    #     num_heads=2,
    #     optimizer="sgd",
    #     lr_schedule="cosine",
    # )
    # + gen_experim(
    #     32,
    #     label="long warmup",
    #     folder_name="variation_experiments",
    #     learning_rate=1e-1,
    #     num_heads=2,
    #     optimizer="sgd",
    #     warmup_frac=0.1,
    # )
)


# GRAND_EXPERIMENT = (
#     TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR_NO_ROTARY
#     + create_multi_lr_experiments(
#         TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR_NO_ROTARY, NARROW_LR_SWEEP
#     )
# )

# do sgd and rmsprop lr tunes

# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
# ) + create_multi_lr_experiments(
#     TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP, NARROW_LR_SWEEP
# )

# rms prop and sgd scaling
# GRAND_EXPERIMENT = (
#     TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR
#     + TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP
# )

# GRAND_EXPERIMENT = (
#     gen_experim(
#         64,
#         label="64d_sgd_cosine_annealing",
#         folder_name="variation_experiments",
#         learning_rate=1e-1,
#         num_heads=2,
#         optimizer="sgd",
#         lr_schedule="cosine",
#     )
#     + gen_experim(
#         64,
#         label="64d_sgd_cosine_annealing_small_lr",
#         folder_name="variation_experiments",
#         learning_rate=1e-2,
#         num_heads=2,
#         optimizer="sgd",
#         lr_schedule="cosine",
#     )
#     + gen_experim(
#         64,
#         label="64d_low_lr",
#         folder_name="variation_experiments",
#         learning_rate=1e-2,
#         num_heads=2,
#         optimizer="sgd",
#     )
#     + gen_experim(
#         64,
#         label="64d_10_1.5_lr",
#         folder_name="variation_experiments",
#         learning_rate=10 ** (-1.5),
#         num_heads=2,
#         optimizer="sgd",
#     )
# )


# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
# )


BEST_POSSIBLE_SGD_SCALING = (
    gen_experim(
        32,
        label="32d_best_sgd",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        effective_batch_size=64,
    )
    + gen_experim(
        48,
        label="48d_best_sgd",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        effective_batch_size=64,
    )
    + gen_experim(
        64,
        label="64d_best_sgd",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        effective_batch_size=64,
    )
)


# GRAND_EXPERIMENT = (
#     TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR
#     + create_multi_lr_experiments(best_sgd, NARROW_LR_SWEEP)
# )

GRAND_EXPERIMENT = BEST_POSSIBLE_SGD_SCALING
