from experiment_utils import (
    create_multi_seed_experiments,
    create_multi_lr_experiments,
    calculate_transformer_params,
    gen_experim,
    get_base_config,
)

LR_ADHOC = [10 ** (-4), 10 ** (-3.5)]
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
SGD_LR_SWEEP = [
    10 ** (-0.5),
    10 ** (-1),
    10 ** (-1.5),
]
SEEDS = [123, 456, 789, 101]
TRANSFORMER_SWEEP = [10 ** (-2.5), 10 ** (-2), 10 ** (-1.5)]


# optimizer variation experiments ?
# positional encoding experiments
# norm experiments
# lr_schedule expeiremtns
# initalizaiton experiments


# gen_experim(
#     32,
#     label="32d_best_sgdbs64lr1",
#     folder_name="sgd_scaling",
#     learning_rate=1e-1,
#     optimizer="sgd",
#     sgd_momentum=0.98,
#     weight_decay=0.0,
#     target_effective_batch_size=64,
#     grad_clip=0.75,
#     warmup_frac=0.1,
# )
# + gen_experim(
#     48,
#     label="48d_best_sgdbs64lr1",
#     folder_name="best_possible_sgd",
#     learning_rate=1e-1,
#     optimizer="sgd",
#     sgd_momentum=0.98,
#     weight_decay=0.0,
#     target_effective_batch_size=64,
#     grad_clip=0.75,
#     warmup_frac=0.1,
# )
# + gen_experim(
#     64,
#     label="64d_best_sgdbs64lr1",
#     folder_name="best_possible_sgd",
#     learning_rate=1e-1,
#     optimizer="sgd",
#     sgd_momentum=0.98,
#     weight_decay=0.0,
#     target_effective_batch_size=64,
#     grad_clip=0.75,
#     warmup_frac=0.1,
# )
# + gen_experim(
#     80,
#     label="80d_best_sgdbs64lr1",
#     folder_name="best_possible_sgd",
#     learning_rate=10 ** (-1),
#     optimizer="sgd",
#     sgd_momentum=0.98,
#     weight_decay=0.0,
#     target_effective_batch_size=64,
#     grad_clip=0.75,
#     warmup_frac=0.1,
# )

SGD_SCALING = (
    gen_experim(
        32,
        label="32d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        48,
        label="48d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        64,
        label="64d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    # setu 96 128 160 192 224 256
    + gen_experim(
        96,
        label="96d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        128,
        label="128d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        160,
        label="160d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        192,
        label="192d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        256,
        label="256d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-0.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
)

LARGE_SCALE_EXPERIMENTS = (
    gen_experim(
        128,
        label="128d_best_sgdbs64",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=64,
    )
    + gen_experim(
        256,
        label="256d_best_sgdbs64",
        folder_name="best_possible_sgd",
        learning_rate=1e-1,
        optimizer="sgd",
        sgd_momentum=0.98,
        weight_decay=0.0,
        target_effective_batch_size=64,
    )
    # sinusoidal experiments
    + gen_experim(
        128,
        label="128d_sinbs128",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
        target_effective_batch_size=128,
    )
    + gen_experim(
        256,
        label="256d_sinbs128",
        folder_name="sin_scaling",
        learning_rate=10 ** (-3),
        target_effective_batch_size=128,
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


# TRANSFORMER_SCALING = (
#     gen_experim(
#         32,
#         label="32d_bs128lr2",
#         folder_name="transformer_scaling",
#         learning_rate=10 ** (-2),
#         target_effective_batch_size=128,
#     )
#     + gen_experim(
#         48,
#         label="48d_bs128lr2",
#         folder_name="transformer_scaling",
#         learning_rate=10 ** (-2),
#         target_effective_batch_size=128,
#     )
#     + gen_experim(
#         64,
#         label="64d_bs128lr2",
#         folder_name="transformer_scaling",
#         learning_rate=10 ** (-2),
#         target_effective_batch_size=128,
#     )
#     + gen_experim(
#         80,
#         label="80d_bs128lr25",
#         folder_name="transformer_scaling",
#         learning_rate=10 ** (-2.5),
#         target_effective_batch_size=128,
#     )
# )


TRANSFORMER_SCALING = (
    gen_experim(
        32,
        label="swiglu_32d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2),
    )
    + gen_experim(
        48,
        label="swiglu_48d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2),
    )
    + gen_experim(
        64,
        label="swiglu_64d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
    )
    # setu 96 128 160 192 224 256
    + gen_experim(
        96,
        label="swiglu_96d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        128,
        label="swiglu_128d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        160,
        label="swiglu_160d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        192,
        label="swiglu_192d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        256,
        label="swiglu_256d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
    )
)

SIN_SCALING = (
    gen_experim(
        32,
        label="32d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        48,
        label="48d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        64,
        label="64d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    # setu 96 128 160 192 224 256
    + gen_experim(
        96,
        label="96d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
    )
    + gen_experim(
        128,
        label="128d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        160,
        label="160d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        192,
        label="192d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
    )
    + gen_experim(
        256,
        label="256d_sin_bs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2.5),
    )
)


NEW_SCALING_NO_ROTARY = (
    gen_experim(
        32,
        label="32d_sinbs64",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        48,
        label="48d_sinbs64lr2",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        64,
        label="64d_sinbs64lr2",
        folder_name="sin_scaling",
        learning_rate=10 ** (-2),
        pos_encoding="sinusoidal",
    )
    # + gen_experim(
    #     80,
    #     label="80d_sinbs64lr2.5",
    #     folder_name="sin_scaling",
    #     learning_rate=10 ** (-2.5),
    #     pos_encoding="sinusoidal",
    # )
    # + gen_experim(
    #     80,
    #     label="80d_sinbs128lr2",
    #     folder_name="sin_scaling",
    #     learning_rate=10 ** (-2),
    #     pos_encoding="sinusoidal",
    # )
    # + gen_experim(
    #     32,
    #     label="32d_sinbs64lr2",
    #     folder_name="sin_scaling",
    #     learning_rate=10 ** (-2),
    #     target_effective_batch_size=64,
    #     pos_encoding="sinusoidal",
    # )
    # + gen_experim(
    #     48,
    #     label="48d_sinbs64lr2",
    #     folder_name="new_scaling",
    #     learning_rate=10 ** (-2),
    #     target_effective_batch_size=64,
    #     pos_encoding="sinusoidal",
    # )
    # + gen_experim(
    #     64,
    #     label="64d_sinbs64lr2.5",
    #     folder_name="sin_scaling",
    #     learning_rate=10 ** (-2.5),
    #     target_effective_batch_size=64,
    #     pos_encoding="sinusoidal",
    # )
    # + gen_experim(
    #     80,
    #     label="80d_sinbs64",
    #     folder_name="sin_scaling",
    #     learning_rate=10 ** (-2.5),
    #     target_effective_batch_size=64,
    #     pos_encoding="sinusoidal",
    # )
)

BATCH_SIZE_TEST = (
    gen_experim(
        32,
        label="32d_best_sgd_bs32_lr1.5",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
        target_effective_batch_size=16,
        grad_clip=0.75,
        warmup_frac=0.1,
    )
    + gen_experim(
        64,
        label="32d_best_sgd_bs16_lr1.5",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
        target_effective_batch_size=32,
        grad_clip=0.75,
        warmup_frac=0.1,
    )
    # + gen_experim(
    #     64,
    #     label="32d_best_sgd_bs64_lr1.5",
    #     folder_name="sgd_scaling",
    #     learning_rate=10 ** (-1.5),
    #     optimizer="sgd",
    #     sgd_momentum=0.98,
    #     target_effective_batch_size=64,
    #     grad_clip=0.75,
    #     warmup_frac=0.1,
    # )
    # + gen_experim(
    #     64,
    #     label="32d_best_sgd_bs128_lr1.5",
    #     folder_name="sgd_scaling",
    #     learning_rate=10 ** (-1.5),
    #     optimizer="sgd",
    #     sgd_momentum=0.98,
    #     target_effective_batch_size=128,
    #     grad_clip=0.75,
    #     warmup_frac=0.1,
    # )
    # + gen_experim(
    #     64,
    #     label="32d_best_sgd_bs256_lr1.5",
    #     folder_name="sgd_scaling",
    #     learning_rate=10 ** (-1.5),
    #     optimizer="sgd",
    #     sgd_momentum=0.98,
    #     target_effective_batch_size=256,
    #     grad_clip=0.75,
    #     warmup_frac=0.1,
    # )
    # + gen_experim(
    #     32,
    #     label="32d_new_scaling_bs16_lr2.5",
    #     folder_name="new_scaling",
    #     learning_rate=10 ** (-2.5),
    #     target_effective_batch_size=16,
    # )
    # + gen_experim(
    #     32,
    #     label="32d_new_scaling_no_rotary_bs16_lr2.5",
    #     folder_name="new_scaling",
    #     learning_rate=10 ** (-2.5),
    #     pos_encoding="sinusoidal",
    #     target_effective_batch_size=16,
    # )
    # + gen_experim(
    #     32,
    #     label="32d_best_sgd_bs16_lr1",
    #     folder_name="sgd_scaling",
    #     learning_rate=10 ** (-1),
    #     optimizer="sgd",
    #     sgd_momentum=0.98,
    #     target_effective_batch_size=16,
    # )
    # + gen_experim(
    #     32,
    #     label="32d_new_scaling_bs16_lr2",
    #     folder_name="new_scaling",
    #     learning_rate=10 ** (-2),
    #     target_effective_batch_size=16,
    # )
    # + gen_experim(
    #     32,
    #     label="32d_new_scaling_no_rotary_bs16_lr2",
    #     folder_name="new_scaling",
    # )
    # + gen_experim(
    #     32,
    #     label="32d_best_sgd_bs32_lr1",
    #     folder_name="sgd_scaling",
    #     learning_rate=10 ** (-1),
    #     optimizer="sgd",
    #     sgd_momentum=0.98,
    #     target_effective_batch_size=32,
    # )
    # + gen_experim(
    #     32,
    #     label="32d_new_scaling_bs32_lr2",
    #     folder_name="new_scaling",
    #     learning_rate=10 ** (-2),
    #     target_effective_batch_size=32,
    # )
)
# SGD stablization experiments
SGD_STABILIZATION_EXPERIMENTS = gen_experim(
    64,
    label="64d_sgd_stab_bs16",
    folder_name="sgd_stabilization",
    learning_rate=10 ** (-1),
    target_effective_batch_size=16,
    grad_clip=0.75,
    warmup_frac=0.1,
) + gen_experim(
    64,
    label="64d_sgd_stab_bs64",
    folder_name="sgd_stabilization",
    learning_rate=10 ** (-1),
    target_effective_batch_size=64,
    grad_clip=0.75,
    warmup_frac=0.1,
)


# GRAND_EXPERIMENT = create_multi_lr_experiments(NEW_SCALING_NO_ROTARY, NARROW_LR_SWEEP)
# GRAND_EXPERIMENT = create_multi_lr_experiments(SGD_SCALING, SGD_LR_SWEEP)
# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     NEW_SCALING, NARROW_LR_SWEEP
# ) + create_multi_lr_experiments(NEW_SCALING_NO_ROTARY, NARROW_LR_SWEEP)
# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(NEW_SCALING, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(NEW_SCALING_NO_ROTARY, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(SGD_SCALING, NARROW_LR_SWEEP)
# )
# GRAND_EXPERIMENT = create_multi_lr_experiments(SGD_SCALING, SGD_LR_SWEEP)


# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     NEW_SCALING, NARROW_LR_SWEEP
# ) + create_multi_lr_experiments(NEW_SCALING_NO_ROTARY, NARROW_LR_SWEEP)
# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(NEW_SCALING, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(NEW_SCALING_NO_ROTARY, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(SGD_SCALING, NARROW_LR_SWEEP)
# )
# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(TRANSFORMER_SCALING, TRANSFORMER_SWEEP)
#     + create_multi_lr_experiments(SGD_SCALING, SGD_LR_SWEEP)
#     + create_multi_lr_experiments(SIN_SCALING, TRANSFORMER_SWEEP)
# )


HISTORICAL_EXPERIMENTS = (
    gen_experim(
        64,
        label="64transformer_2017_bs64",
        folder_name="historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        64,
        label="64transformer_2022_bs64",
        folder_name="historical_experiments",
        learning_rate=10 ** (-2),
        activation="swiglu",
        norm_type="rms",
    )
    + gen_experim(
        128,
        label="128transformer_2017_bs64",
        folder_name="historical_experiments",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        160,
        label="160transformer_2022_bs64",
        folder_name="historical_experiments",
        learning_rate=10 ** (-2.5),
        activation="swiglu",
        norm_type="rms",
    )
)


# HISTORICAL SCALING


HISTORICAL_EXPERIMENTS_SCALING = (
    # gen_experim(
    #     32,
    #     label="p32transformer_2017_bs64",
    #     folder_name="historical_experiments",
    #     learning_rate=10 ** (-2.5),
    #     activation="gelu",
    #     norm_placement="post",
    #     lr_schedule="inverse_sqrt",
    #     pos_encoding="sinusoidal",
    # )
    gen_experim(
        48,
        label="p48transformer_2017_bs64",
        folder_name="historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    # + gen_experim(
    #     64,
    #     label="p64transformer_2017_bs64",
    #     folder_name="historical_experiments",
    #     learning_rate=10 ** (-3),
    #     activation="gelu",
    #     norm_placement="post",
    #     lr_schedule="inverse_sqrt",
    #     pos_encoding="sinusoidal",
    # )
    + gen_experim(
        80,
        label="p80transformer_2017_bs64",
        folder_name="historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        96,
        label="p96transformer_2017_bs64",
        folder_name="historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    # + gen_experim(
    #     128,
    #     label="p128transformer_2017_bs64",
    #     folder_name="historical_experiments",
    #     learning_rate=10 ** (-3.5),
    #     activation="gelu",
    #     norm_placement="post",
    #     lr_schedule="inverse_sqrt",
    #     pos_encoding="sinusoidal",
    # )
    # + gen_experim(
    #     160,
    #     label="p160transformer_2017_bs64",
    #     folder_name="historical_experiments",
    #     learning_rate=10 ** (-3.5),
    #     activation="gelu",
    #     norm_placement="post",
    #     lr_schedule="inverse_sqrt",
    #     pos_encoding="sinusoidal",
    # )
)
HISTORICAL_EXPERIMENTS_SCALING_LR_TUNE = [10 ** (-2.5), 10 ** (-3), 10 ** (-3.5)]

# GRAND_EXPERIMENT = SIN_SCALING + SGD_SCALING + TRANSFORMER_SCALING
# GRAND_EXPERIMENT = create_multi_lr_experiments(HISTORICAL_EXPERIMENTS, LR_ADHOC)
GRAND_EXPERIMENT = TRANSFORMER_SCALING
