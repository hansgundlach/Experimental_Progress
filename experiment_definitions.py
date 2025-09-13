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
    1e-1,
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
        learning_rate=1e-1,
    )
    + gen_experim(
        40,
        label="vanilla_40d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        48,
        label="vanilla_48d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        56,
        label="vanilla_56d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        64,
        label="vanilla_64d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        80,
        label="vanilla_80d_no_rot",
        folder_name="vanilla_scaling_no_rotary",
        learning_rate=10 ** (-1),
    )
)


TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP = (
    gen_experim(
        32,
        label="vanilla_32d",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=1e-1,
    )
    + gen_experim(
        40,
        label="vanilla_40d",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        48,
        label="vanilla_48d",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        56,
        label="vanilla_56d",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        64,
        label="vanilla_64d",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-1),
    )
    + gen_experim(
        80,
        label="vanilla_80d",
        folder_name="vanilla_scaling_rmsprop",
        learning_rate=10 ** (-1),
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
    gen_experim(
        32,
        label="optimal_lr_sgd_32d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_experim(
        40,
        label="optimal_lr_sgd_40d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_experim(
        48,
        label="optimal_lr_sgd_48d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_experim(
        56,
        label="optimal_lr_sgd_56d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_experim(
        64,
        label="optimal_lr_sgd_64d",
        folder_name="optimal_lr_sgd_scaling",
        learning_rate=0.1,
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

GRAND_EXPERIMENT = (
    TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR
    + TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR
    + TRANSFORMER_SCALING_EXPERIMENTS_RMSPROP
    + create_multi_lr_experiments(
        TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
    )
)
# 6*5+6
