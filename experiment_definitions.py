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
TRANSFORMER_SWEEP = [10 ** (-3.5), 10 ** (-3), 10 ** (-2.5), 10 ** (-2), 10 ** (-1.5)]


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
        label="swiglu_32d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        48,
        label="swiglu_48d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        64,
        label="swiglu_64d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    # setu 96 128 160 192 224 256
    + gen_experim(
        96,
        label="swiglu_96d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        128,
        label="swiglu_128d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        160,
        label="swiglu_160d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        192,
        label="swiglu_192d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
        optimizer="sgd",
        sgd_momentum=0.98,
    )
    + gen_experim(
        256,
        label="swiglu_256d_sgdbs64",
        folder_name="sgd_scaling",
        learning_rate=10 ** (-1.5),
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
        learning_rate=10 ** (-2),
    )
    # 80
    + gen_experim(
        80,
        label="swiglu_80d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2),
    )
    # setu 96 128 160 192 224 256
    + gen_experim(
        96,
        label="swiglu_96d_transformer_bs64",
        folder_name="transformer_scaling",
        learning_rate=10 ** (-2.5),
    )
    # + gen_experim(
    #     128,
    #     label="swiglu_128d_transformer_bs64",
    #     folder_name="transformer_scaling",
    #     learning_rate=10 ** (-2.5),
    # )
    # + gen_experim(
    #     160,
    #     label="swiglu_160d_transformer_bs64",
    #     folder_name="transformer_scaling",
    #     learning_rate=10 ** (-2.5),
    # )
    # + gen_experim(
    #     192,
    #     label="swiglu_192d_transformer_bs64",
    #     folder_name="transformer_scaling",
    #     learning_rate=10 ** (-2.5),
    # )
    # + gen_experim(
    #     256,
    #     label="swiglu_256d_transformer_bs64",
    #     folder_name="transformer_scaling",
    #     learning_rate=10 ** (-3),
    # )
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


# were these the orginal no scaling experiments
# HISTORICAL_EXPERIMENTS = (
#     gen_experim(
#         64,
#         label="64transformer_2017_bs64",
#         folder_name="historical_experiments",
#         learning_rate=10 ** (-2.5),
#         activation="gelu",
#         norm_placement="post",
#         lr_schedule="inverse_sqrt",
#         pos_encoding="sinusoidal",
#     )
#     + gen_experim(
#         64,
#         label="64transformer_2022_bs64",
#         folder_name="historical_experiments",
#         learning_rate=10 ** (-2.5),
#         activation="swiglu",
#         norm_type="rms",
#     )
#     + gen_experim(
#         128,
#         label="128transformer_2017_bs64",
#         folder_name="historical_experiments",
#         learning_rate=10 ** (-3.5),
#         activation="gelu",
#         norm_placement="post",
#         lr_schedule="inverse_sqrt",
#         pos_encoding="sinusoidal",
#     )
#     + gen_experim(
#         160,
#         label="160transformer_2022_bs64",
#         folder_name="historical_experiments",
#         learning_rate=10 ** (-2.5),
#         activation="swiglu",
#         norm_type="rms",
#     )
# )


# HISTORICAL SCALING


HISTORICAL_EXPERIMENTS_SCALING = (
    gen_experim(
        32,
        label="t_newold_32transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-2.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        48,
        label="t_newold_p48transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        64,
        label="t_newold_64transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        80,
        label="t_newold_p80transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        96,
        label="t_newold_p96transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        128,
        label="t_newold_p128transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        160,
        label="t_newold_p160transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        256,
        label="t_newold_p256transformer_2017_bs64",
        folder_name="retry_historical_experiments",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="inverse_sqrt",
        pos_encoding="sinusoidal",
    )
)


HISTORICAL_EXPERIMENTS_VASWANI = (
    gen_experim(
        32,
        label="vaswani_32transformer_2017_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=1,
        activation="gelu",
        norm_placement="post",
        lr_schedule="transformer",
        pos_encoding="sinusoidal",
        dropout=0.1,
        label_smoothing=0.1,
        optimizer="adam",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
    )
    + gen_experim(
        48,
        label="vaswani_p48transformer_2017_bs64",
        folder_name="debug_historical_experiments",
        activation="gelu",
        norm_placement="post",
        lr_schedule="transformer",
        learning_rate=1,
        pos_encoding="sinusoidal",
        dropout=0.1,
        label_smoothing=0.1,
        optimizer="adam",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
    )
    + gen_experim(
        64,
        label="vaswani_64transformer_2017_bs64",
        folder_name="debug_historical_experiments",
        activation="gelu",
        norm_placement="post",
        lr_schedule="transformer",
        learning_rate=1,
        pos_encoding="sinusoidal",
        dropout=0.1,
        label_smoothing=0.1,
        optimizer="adam",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
    )
)

HISTORICAL_EXPERIMENTS_RADFORD = (
    gen_experim(
        32,
        label="radford_32transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=5.6e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        weight_decay=0.01,
        pos_encoding="learned",
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        48,
        label="radford_48transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=2.9e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        64,
        label="radford_64transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=1.86e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        80,
        label="radford_80transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=1.30e-3,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        96,
        label="radford_96transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=9.79e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    # continue scaling experiments 104, 128, 160, 256
    + gen_experim(
        104,
        label="radford_104transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=8.62e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        128,
        label="radford_128transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=6.1996e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        160,
        label="radford_160transformer_2018_bs64",
        folder_name="debug_historical_experiments",
        learning_rate=4.34e-4,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    # + gen_experim(
    #     256,
    #     label="radford_256transformer_2018_bs64",
    #     folder_name="debug_historical_experiments",
    #     learning_rate=10 ** (-3.5),
    #     activation="gelu",
    #     norm_placement="post",
    #     lr_schedule="linear_warmup",
    #     pos_encoding="learned",
    #     weight_decay=0.01,
    #     dropout=0.0,
    #     optimizer="adam",
    #     modern_bias_0=False,
    #     ff_ratio=4,
    # )
)

HISTORICAL_LR_STUDY = create_multi_lr_experiments(
    gen_experim(
        160,
        label="radford_160_2018_bs64",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        256,
        label="radford_256_2018_bs64",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-3.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
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
        label="radford_32_2018_bs64",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-2.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        weight_decay=0.01,
        pos_encoding="learned",
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        64,
        label="radford_64_2018_bs64",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-2.5),
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        96,
        label="radford_96_2018_bs64",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    ),
    [10**-2.25, 10**-2.5, 10**-2.75, 10**-3, 10**-3.25],
)


HISTORICAL_EXPERIMENTS_SCALING_LR_TUNE = [10 ** (-2.5), 10 ** (-3), 10 ** (-3.5)]

# GRAND_EXPERIMENT = SIN_SCALING + SGD_SCALING + TRANSFORMER_SCALING
# GRAND_EXPERIMENT = create_multi_lr_experiments(HISTORICAL_EXPERIMENTS, LR_ADHOC)
# GRAND_EXPERIMENT = gen_experim(
#     64,
#     label="radford_no_reg",
#     folder_name="debug_historical_experiments",
#     learning_rate=10 ** (-2.5),
#     activation="gelu",
#     norm_placement="post",
#     lr_schedule="linear_warmup",
#     pos_encoding="learned",
#     weight_decay=0.0,
#     dropout=0.0,
#     optimizer="adam",
# )

CURRENT_EXPERIMENT = (
    gen_experim(
        128,
        label="radford_128",
        folder_name="debug_historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
    )
    + gen_experim(
        128,
        label="modern_128",
        folder_name="debug_historical_experiments",
        learning_rate=10 ** (-2.5),
        activation="swiglu",
        norm_type="layer",
        norm_placement="pre",
        lr_schedule="cosine_warmup",
        pos_encoding="rotary",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adamw",
    )
    + gen_experim(
        128,
        label="radford_sin",
        folder_name="debug_historical_experiments",
        learning_rate=10 ** (-3),
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="sinusoidal",
        dropout=0.0,
        weight_decay=0.01,
        optimizer="adam",
    )
)


NEW_VARIATIONS_EXPERIMENTS = (
    gen_experim(
        160,
        label="radford_160transformer_bs64",
        folder_name="new_variations",
        learning_rate=10**-3.5,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        160,
        label="sin_radford_160_bs64",
        folder_name="new_variations",
        learning_rate=10**-3.5,
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,
        dropout=0.0,
        optimizer="adam",
        modern_bias_0=False,
        ff_ratio=4,
    )
    + gen_experim(
        160,
        label="160_modern_rms",
        folder_name="new_variations",
        learning_rate=10**-3,
        modern_bias_0=True,
        norm_type="rms",
        weight_decay=0.01,
        optimizer="adamw",
        ff_ratio=4,
    )
    + gen_experim(
        160,
        label="160_modern_no_rms",
        folder_name="new_variations",
        learning_rate=10**-3,
        modern_bias_0=True,
        norm_type="layer",
        weight_decay=0.01,
        optimizer="adamw",
        ff_ratio=4,
    )
    # position encoding experiments
    # modern experiment
    #
    # gen_experim(
    #     64,
    #     label="old_modern",
    #     folder_name="new_variations",
    #     learning_rate=10**-2,
    #     modern_bias_0=False,
    #     norm_type="layer",
    #     weight_decay=0.0,
    #     optimizer="adam",
    #     ff_ratio=4,
    # )
)

# modern variation scaling study
MODERN_SCALING_STUDY = (
    # 32 48 64 80 96 128 160
    gen_experim(
        32,
        label="32_modern",
        folder_name="new_modern_scaling_study",
        learning_rate=9.95e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        
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
        label="64_modern",
        folder_name="new_modern_scaling_study",
        learning_rate=5.79e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        80,
        label="80_modern",
        folder_name="new_modern_scaling_study",
        learning_rate=4.87e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        96,
        label="96_modern",
        folder_name="new_modern_scaling_study",
        learning_rate=4.23e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        104,
        label="104_modern",
        folder_name="new_modern_scaling_study",
        learning_rate=4.23e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        128,
        label="128_modern",
        folder_name="new_modern_scaling_study",
        learning_rate=3.37e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
    + gen_experim(
        160,
        label="160_modern",
        folder_name="new_modern_scaling_study",
        learning_rate=2.83e-3,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
    )
)

# learning rate study
# modern variation scaling study
LR_STUDY = (
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


# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(
#         gen_experim(
#             96,
#             label="96_modern",
#             folder_name="modern_scaling_study",
#             learning_rate=10**-2.5,
#             modern_bias_0=True,
#             ff_ratio=4,
#             norm_type="rms",
#         ),
#         [10**-2.75],
#     )
#     + create_multi_lr_experiments(
#         gen_experim(
#             160,
#             label="160_modern",
#             folder_name="modern_scaling_study",
#             learning_rate=10**-2.5,
#             modern_bias_0=True,
#             ff_ratio=4,
#             norm_type="rms",
#         ),
#         [10**-2.25],
#     )
#     + create_multi_lr_experiments(
#         gen_experim(
#             128,
#             label="128_modern",
#             folder_name="modern_scaling_study",
#             learning_rate=10**-2.5,
#             modern_bias_0=True,
#             ff_ratio=4,
#             norm_type="rms",
#         ),
#         [10**-2.25, 10**-2.75],
#     )
# )

GRAND_EXPERIMENT = (
    # gen experiment 88 and 104
    # gen_experim(
    #     88,
    #     label="88_modern",
    #     folder_name="modern_scaling_study",
    #     learning_rate=10**-2,
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     norm_type="rms",
    # )
    # + gen_experim(
    #     104,
    #     label="104_modern",
    #     folder_name="modern_scaling_study",
    #     learning_rate=10**-2.5,
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     norm_type="rms",
    # )
    gen_experim(
        160,
        label="160_modern_225",
        folder_name="modern_scaling_study",
        learning_rate=10**-2.25,
        modern_bias_0=True,
        ff_ratio=4,
        norm_type="rms",
    )
    + gen_experim(
        160,
        label="160_modern_275",
        folder_name="modern_scaling_study",
        learning_rate=10**-2.75,
        modern_bias_0=True,
        ff_ratio=4,
        norm_type="rms",
    )
    + gen_experim(
        152,
        label="152_modern_25",
        folder_name="modern_scaling_study",
        learning_rate=10**-2.5,
        modern_bias_0=True,
        ff_ratio=4,
        norm_type="rms",
    )
)

GRAND_EXPERIMENT = (
    gen_experim(
        96,
        label="96_init_try2",
        folder_name="modern_scaling_study",
        learning_rate=10**-2.25,
        modern_bias_0=True,
        ff_ratio=4,
        norm_type="rms",
        init_scheme="transformer_scaled",
    )
    + gen_experim(
        96,
        label="96_modern_ff25",
        folder_name="modern_scaling_study",
        learning_rate=10**-2.25,
        modern_bias_0=True,
        ff_ratio=2.5,
        norm_type="rms",
        init_scheme="transformer_scaled",
    )
    + gen_experim(
        96,
        label="96_modern_layer2",
        folder_name="modern_scaling_study",
        learning_rate=10**-2.25,
        modern_bias_0=True,
        ff_ratio=4,
        norm_type="rms",
        num_layers=2,
    )
)


# setup mup learning rate experiments at hidden dimension 64, 96, 128, 160
# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     gen_experim(
#         64,
#         label="64_modern_mup",
#         folder_name="modern_scaling_study",
#         learning_rate=10**-1,
#         modern_bias_0=True,
#         ff_ratio=4,
#         norm_type="rms",
#         use_mup=True,
#         mup_base_width=64,
#     ),
#     NARROW_LR_SWEEP,
# )

# set up series of mup experiments from 64 to 160
# GRAND_EXPERIMENT = (
#     gen_experim(
#         64,
#         label="64_modern_mup",
#         folder_name="modern_scaling_study",
#         learning_rate=10**-1,
#         modern_bias_0=True,
#         ff_ratio=4,
#         norm_type="rms",
#         use_mup=True,
#         mup_base_width=64,
#     )
#     + gen_experim(
#         96,
#         label="96_modern_mup",
#         folder_name="modern_scaling_study",
#         learning_rate=10**-1,
#         modern_bias_0=True,
#         ff_ratio=4,
#         norm_type="rms",
#         use_mup=True,
#         mup_base_width=64,
#     )
#     + gen_experim(
#         128,
#         label="128_modern_mup",
#         folder_name="modern_scaling_study",
#         learning_rate=10**-1,
#         modern_bias_0=True,
#         ff_ratio=4,
#         norm_type="rms",
#         use_mup=True,
#         mup_base_width=64,
#     )
#     + gen_experim(
#         160,
#         label="160_modern_mup",
#         folder_name="modern_scaling_study",
#         learning_rate=10**-1,
#         modern_bias_0=True,
#         ff_ratio=4,
#         norm_type="rms",
#         use_mup=True,
#         mup_base_width=64,
#     )
# )

# gen_experim(
#     96,
#     label="96_modern_15",
#     folder_name="modern_scaling_study",
#     learning_rate=10**-2,
#     modern_bias_0=True,
#     ff_ratio=4,
#     norm_type="rms",
# )
# gen_experim(
#     96,
#     label="96_modern_layer_init",
#     folder_name="modern_scaling_study",
#     learning_rate=10**-2,
#     modern_bias_0=True,
#     ff_ratio=4,
#     norm_type="rms",
#     init_scheme="transformer_scaled",
# )
# + gen_experim(
#     96,
#     label="96_modern_no_weight_decay",
#     folder_name="modern_scaling_study",
#     learning_rate=10**-2,
#     modern_bias_0=True,
#     ff_ratio=4,
#     norm_type="rms",
#     weight_decay=0.00,
# )
# + gen_experim(
#     96,
#     label="96_modern_no_weight_decay",
#     folder_name="modern_scaling_study",
#     learning_rate=10**-2,
#     modern_bias_0=True,
#     ff_ratio=4,
#     norm_type="rms",
#     weight_decay=0.00,
# )
# + gen_experim(
#     96,
#     label="96_modern_275",
#     folder_name="modern_scaling_study",
#     learning_rate=10**-2.75,
#     modern_bias_0=True,
#     ff_ratio=4,
#     norm_type="rms",
# )
# + gen_experim(
#     96,
#     label="96_modern_3",
#     folder_name="modern_scaling_study",
#     learning_rate=10**-3,
#     modern_bias_0=True,
#     ff_ratio=4,
#     norm_type="rms",
# )

# GRAND_EXPERIMENT = gen_experim(
#     96,
#     label="96_modern_15",
#     folder_name="modern_scaling_study",
#     learning_rate=10**-1.5,
#     modern_bias_0=True,
#     ff_ratio=4,
#     norm_type="rms",
# )

# GRAND_EXPERIMENT = HISTORICAL_EXPERIMENTS_RADFORD


# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     LR_STUDY, [10**-2.75, 10**-2.5, 10**-2.25, 10**-2, 10**-1.75]
# )


# great width 256
#
GRAND_EXPERIMENT = HISTORICAL_EXPERIMENTS_RADFORD
