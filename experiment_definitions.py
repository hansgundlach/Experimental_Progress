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
VERY_NARROW_LR_SWEEP = [
    10 ** (-2),
    10 ** (-1.5),
    10 ** (-1),
]


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


SGD_SCALING = (
    gen_experim(
        32,
        label="32d_best_sgdbs32",
        folder_name="sgd_scaling",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=32,
    )
    + gen_experim(
        48,
        label="48d_best_sgdbs32",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=32,
    )
    + gen_experim(
        64,
        label="64d_best_sgdbs32",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=32,
    )
    + gen_experim(
        80,
        label="80d_best_sgdbs32",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=32,
    )
)


SGD_SCALING_LARGE_BATCH = (
    gen_experim(
        32,
        label="32d_best_sgdbs128lr1",
        folder_name="sgd_scaling",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=128,
    )
    + gen_experim(
        48,
        label="48d_best_sgdbs128lr1",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=128,
    )
    + gen_experim(
        64,
        label="64d_best_sgdbs128lr1",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=128,
    )
    + gen_experim(
        80,
        label="80d_best_sgdbs128lr1",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=128,
    )
)


# BATCH_SIZE_TEST = gen_experim(
#     32,
#     label="32d_sgd_bs16",
#     folder_name="sgd_scaling",
#     learning_rate=1e-1,
#     optimizer="sgd",
#     sgd_momentum=0.98,
#     weight_decay=0.0,
#     target_effective_batch_size=16,
# )
# +
# BATCH_SIZE_TEST = gen_experim(
#     32,
#     label="32d_sgd_bs128",
#     folder_name="sgd_scaling",
#     learning_rate=1e-1,
#     optimizer="sgd",
#     sgd_momentum=0.98,
#     weight_decay=0.0,
#     target_effective_batch_size=128,
# )


NEW_SCALING = (
    gen_experim(
        32,
        label="32d_bs128lr2",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=128,
    )
    + gen_experim(
        48,
        label="48d_bs128lr2",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=128,
    )
    + gen_experim(
        64,
        label="64d_bs128lr2",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=128,
    )
    + gen_experim(
        80,
        label="80d_bs128lr25",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
        target_effective_batch_size=128,
    )
)


NEW_SCALING_NO_ROTARY = (
    gen_experim(
        32,
        label="32d_sinbs128",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=128,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        48,
        label="48d_sinbs128lr2",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=128,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        64,
        label="64d_sinbs128lr2",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=128,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        80,
        label="80d_sinbs128lr2.5",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
        target_effective_batch_size=128,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        80,
        label="80d_sinbs128lr2",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=128,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        32,
        label="32d_sinbs64lr2",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=64,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        48,
        label="48d_sinbs64lr2",
        folder_name="new_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=64,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        64,
        label="64d_sinbs64lr2.5",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
        target_effective_batch_size=64,
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        80,
        label="80d_sinbs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
        target_effective_batch_size=64,
        pos_encoding="sinusoidal",
    )
)

BATCH_SIZE_TEST = (
    gen_experim(
        32,
        label="32d_best_sgd_bs16_lr1.5",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
        target_effective_batch_size=16,
    )
    + gen_experim(
        32,
        label="32d_new_scaling_bs16_lr2.5",
        folder_name="new_scaling",
        learning_rate=10 ** (-2.5),
        target_effective_batch_size=16,
    )
    + gen_experim(
        32,
        label="32d_new_scaling_no_rotary_bs16_lr2.5",
        folder_name="new_scaling",
        learning_rate=10 ** (-2.5),
        pos_encoding="sinusoidal",
        target_effective_batch_size=16,
    )
    + gen_experim(
        32,
        label="32d_best_sgd_bs16_lr1",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1),
        optimizer="sgd",
        sgd_momentum=0.98,
        target_effective_batch_size=16,
    )
    + gen_experim(
        32,
        label="32d_new_scaling_bs16_lr2",
        folder_name="new_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=16,
    )
    + gen_experim(
        32,
        label="32d_new_scaling_no_rotary_bs16_lr2",
        folder_name="new_scaling",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
        target_effective_batch_size=16,
    )
    + gen_experim(
        32,
        label="32d_best_sgd_bs32_lr1",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1),
        optimizer="sgd",
        sgd_momentum=0.98,
        target_effective_batch_size=32,
    )
    + gen_experim(
        32,
        label="32d_new_scaling_bs32_lr2",
        folder_name="new_scaling",
        learning_rate=10 ** (-2),
        target_effective_batch_size=32,
    )
)

GRAND_EXPERIMENT = NEW_SCALING

# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     NEW_SCALING, NARROW_LR_SWEEP
# ) + create_multi_lr_experiments(NEW_SCALING_NO_ROTARY, NARROW_LR_SWEEP)
# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(NEW_SCALING, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(NEW_SCALING_NO_ROTARY, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(SGD_SCALING, NARROW_LR_SWEEP)
# )
