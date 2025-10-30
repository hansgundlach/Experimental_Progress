from experiment_utils import (
    create_multi_seed_experiments,
    create_multi_lr_experiments,
    calculate_transformer_params,
    gen_experim,
    get_base_config,
)

NARROW_LR_SWEEP = [
    10 ** (-2.5),
    10 ** (-2),
    10 ** (-1.5),
]
DIMENSION = 64
SEEDS = [123, 456, 789, 101]
folder_name = "stanford_mult"
learning_rate = 5.798e-3

# activation variation experiments
ACTIVATION_EXPERIMENTS = gen_experim(
    DIMENSION,
    label=f"{DIMENSION}d_gelu",
    learning_rate=learning_rate,
    activation="gelu",
    folder_name=folder_name,
) + gen_experim(
    DIMENSION,
    label=f"{DIMENSION}d_relu",
    learning_rate=learning_rate,
    activation="relu",
    folder_name=folder_name,
)

# optimizer variation experiments ?
# positional encoding experiments
# norm experiments
# lr_schedule expeiremtns
# initalizaiton experiments
INITIALIZATION_EXPERIMENTS = (
    # gen_experim(
    #     DIMENSION,
    #     label="64d_xavier_normal",
    #     learning_rate=0.01,
    #     init_scheme="xavier_normal",
    #     folder_name="alg_mult",
    # )
    # + gen_experim(
    #     DIMENSION,
    #     label="64d_kaiming_normal",
    #     learning_rate=0.01,
    #     init_scheme="kaiming_normal",
    #     folder_name="alg_mult",
    # )
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_transformer_scaled_init",
        learning_rate=learning_rate,
        init_scheme="transformer_scaled",
        folder_name=folder_name,
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_removing_bias",
        learning_rate=learning_rate,
        init_scheme="bert_gpt",
        folder_name=folder_name,
        modern_bias_0=False,
    )
)


# normalization experiments
NORMALIZATION_EXPERIMENTS = (
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_layer",
        folder_name=folder_name,
        learning_rate=learning_rate,
        norm_type="layer",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_rms",
        folder_name=folder_name,
        learning_rate=learning_rate,
        norm_type="rms",
    )
    # add no norm placement option
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_post",
        folder_name=folder_name,
        learning_rate=1e-3,
        norm_placement="post",
    )
)

OPTIMIZER_EXPERIMENTS = (
    # + gen_experim(
    #     DIMENSION,
    #     label=f"{DIMENSION}d_sgd",
    #     folder_name=folder_name,
    #     learning_rate=learning_rate,
    #     optimizer="sgd",
    # )
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_adam",
        folder_name=folder_name,
        learning_rate=learning_rate,
        optimizer="adam",
    )
)


POSITIONAL_ENCODING_EXPERIMENTS = gen_experim(
    DIMENSION,
    label=f"{DIMENSION}d_sinusoidal",
    folder_name=folder_name,
    learning_rate=learning_rate,
    pos_encoding="sinusoidal",
) + gen_experim(
    DIMENSION,
    label=f"{DIMENSION}d_learned",
    folder_name=folder_name,
    learning_rate=learning_rate,
    pos_encoding="learned",
)

# you would probably have to do an lr tune for the linear warmup
LR_SCHEDULE_EXPERIMENTS = (
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_linear_warmup",
        folder_name=folder_name,
        learning_rate=learning_rate,
        lr_schedule="linear_warmup",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_lr_inverse_sqrt",
        folder_name=folder_name,
        learning_rate=learning_rate,
        lr_schedule="inverse_sqrt",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_lr_transformer",
        folder_name=folder_name,
        learning_rate=learning_rate,
        lr_schedule="transformer",
    )
)

LR_SWEEP = [10 ** (-2.5), 10 ** (-2), 10 ** (-1.5)]

# post layer norm lr tune necessary
# GRAND_VARIATION_EXPERIMENTS = create_multi_lr_experiments(
#     gen_experim(
#         DIMENSION,
#         label=f"{DIMENSION}d_post",
#         folder_name=folder_name,
#         learning_rate=learning_rate,
#         norm_placement="post",
#     )
#     + gen_experim(
#         DIMENSION,
#         label=f"{DIMENSION}d_layer",
#         folder_name="alg_mult",
#         learning_rate=learning_rate,
#         norm_type="layer",
#     ),
#     [
#         10 ** (-3.5),
#         10 ** (-3),
#         10 ** (-2.5),
#         10 ** (-2),
#         10 ** (-1.5),
#         10 ** (-1),
#     ],
# )

# 16 experiments
# learning rate experimetns
# GRAND_VARIATION_EXPERIMENTS = (
#     create_multi_lr_experiments(LR_SCHEDULE_EXPERIMENTS, LR_SWEEP)
#     + create_multi_lr_experiments(INITIALIZATION_EXPERIMENTS, LR_SWEEP)
#     + create_multi_lr_experiments(NORMALIZATION_EXPERIMENTS, LR_SWEEP)
#     + create_multi_lr_experiments(POSITIONAL_ENCODING_EXPERIMENTS, LR_SWEEP)
#     + create_multi_lr_experiments(ACTIVATION_EXPERIMENTS, LR_SWEEP)
# )


GRAND_VARIATION_EXPERIMENTS = (
    LR_SCHEDULE_EXPERIMENTS
    + INITIALIZATION_EXPERIMENTS
    + NORMALIZATION_EXPERIMENTS
    + POSITIONAL_ENCODING_EXPERIMENTS
    + ACTIVATION_EXPERIMENTS
    + OPTIMIZER_EXPERIMENTS
)


HISTORICAL_ORDER = (
    # 0) 2017 ORIGINAL TRANSFORMER BASELINE
    gen_experim(
        64,
        label="00_base_2017_post_noam_relu_sinus",
        folder_name="historical_lr_study",
        learning_rate=10 ** (-2.25),  # tune once and freeze across steps
        activation="relu",  # 2017
        norm_placement="post",  # Post-LN
        lr_schedule="inverse_sqrt",  # Noam
        pos_encoding="sinusoidal",  # 2017 default
        weight_decay=0.0,  # Adam + L2 effectively off in 2017
        dropout=0.1,  # typical in 2017
        optimizer="adam",
        modern_bias_0=True,  # bias=0 has long been common
        ff_ratio=4,
    )
    # 1) 2017 ALTERNATIVE: LEARNED PE
    # + gen_experim(
    #     64,
    #     label="01_learned_pe",
    #     folder_name="historical_lr_study",
    #     activation="relu",
    #     norm_placement="post",
    #     lr_schedule="inverse_sqrt",
    #     pos_encoding="learned",  # only change
    #     weight_decay=0.0,
    #     dropout=0.1,
    #     optimizer="adam",
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     learning_rate=10 ** (-2.5),
    # )
    # # 2) 2018 ADOPTION: GELU (keep schedule/PE from previous step)
    # + gen_experim(
    #     64,
    #     label="02_gelu_activation",
    #     folder_name="historical_lr_study",
    #     activation="gelu",  # change
    #     norm_placement="post",
    #     lr_schedule="inverse_sqrt",
    #     pos_encoding="learned",
    #     weight_decay=0.0,
    #     dropout=0.1,
    #     optimizer="adam",
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     learning_rate=10 ** (-2.75),
    # )
    # 3) 2018–2019: LINEAR DECAY (with warmup)
    + gen_experim(
        64,
        label="03_linear_decay",
        folder_name="historical_lr_study",
        activation="gelu",
        norm_placement="post",
        lr_schedule="linear_warmup",  # change (warmup + linear → 0)
        pos_encoding="learned",
        weight_decay=0.0,
        dropout=0.1,
        optimizer="adam",
        modern_bias_0=True,
        ff_ratio=4,
        learning_rate=10 ** (-2.5),
    )
    # 4) ~2019: PRE-LN (stability)
    + gen_experim(
        64,
        label="04_pre_ln",
        folder_name="historical_lr_study",
        activation="gelu",
        norm_placement="pre",  # change
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.0,
        dropout=0.1,
        optimizer="adam",
        modern_bias_0=True,
        ff_ratio=4,
        learning_rate=10 ** (-2.5),
    )
    # 5) ~2019: ADAMW (decoupled weight decay)
    + gen_experim(
        64,
        label="05_adamw",
        folder_name="historical_lr_study",
        activation="gelu",
        norm_placement="pre",
        lr_schedule="linear_warmup",
        pos_encoding="learned",
        weight_decay=0.01,  # change (now meaningful)
        dropout=0.1,
        optimizer="adamw",  # change
        modern_bias_0=True,
        ff_ratio=4,
        learning_rate=10 ** (-2.5),
    )
    # 6) 2019–2020: COSINE DECAY (no restarts)
    # + gen_experim(
    #     64,
    #     label="06_cosine_decay",
    #     folder_name="historical_lr_study",
    #     activation="gelu",
    #     norm_placement="pre",
    #     lr_schedule="cosine_warmup",  # change
    #     pos_encoding="learned",
    #     weight_decay=0.01,
    #     dropout=0.1,
    #     optimizer="adamw",
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     learning_rate=10 ** (-2.25),
    # )
    # 7) 2019–2020: RMSNorm
    # + gen_experim(
    #     64,
    #     label="07_rmsnorm",
    #     folder_name="historical_lr_study",
    #     activation="gelu",
    #     norm_placement="pre",
    #     lr_schedule="cosine_warmup",
    #     pos_encoding="learned",
    #     weight_decay=0.01,
    #     dropout=0.1,
    #     optimizer="adamw",
    #     norm_type="rms",  # change
    #     modern_bias_0=True,
    #     ff_ratio=4,
    #     learning_rate=10 ** (-2.25),
    # )
    # # 8) 2020: SwiGLU (+ typical reduced FF width)
    # + gen_experim(
    #     64,
    #     label="08_swiglu_ff2p5",
    #     folder_name="historical_lr_study",
    #     activation="swiglu",  # change
    #     norm_placement="pre",
    #     lr_schedule="cosine_warmup",
    #     pos_encoding="learned",
    #     weight_decay=0.01,
    #     dropout=0.1,
    #     optimizer="adamw",
    #     norm_type="rms",
    #     modern_bias_0=True,
    #     ff_ratio=2.5,  # change (SwiGLU often uses ~2.5–3.0)
    #     learning_rate=10 ** (-2),
    # )
    # # 9) 2021: RoPE
    # + gen_experim(
    #     64,
    #     label="09_rope",
    #     folder_name="historical_lr_study",
    #     activation="swiglu",
    #     norm_placement="pre",
    #     lr_schedule="cosine_warmup",
    #     pos_encoding="rotary",  # change
    #     weight_decay=0.01,
    #     dropout=0.1,
    #     optimizer="adamw",
    #     norm_type="rms",
    #     modern_bias_0=True,
    #     ff_ratio=2.5,
    #     learning_rate=10 ** (-2),
    # )
    # # 10) 2022+: Remove dropout in pretraining
    # + gen_experim(
    #     64,
    #     label="10_no_dropout",
    #     folder_name="historical_lr_study",
    #     activation="swiglu",
    #     norm_placement="pre",
    #     lr_schedule="cosine_warmup",
    #     pos_encoding="rotary",
    #     weight_decay=0.01,
    #     dropout=0.0,  # change
    #     optimizer="adamw",
    #     norm_type="rms",
    #     modern_bias_0=True,
    #     ff_ratio=2.5,
    #     learning_rate=10 ** (-2),
    # )
)


# 5* 11 = 55

HISTORICAL_ORDER_LR_TUNE = create_multi_lr_experiments(
    HISTORICAL_ORDER, [10**-3, 10**-2.75, 10**-2.5, 10**-2.25, 10**-2]
)


GRAND_VARIATION_EXPERIMENTS = create_multi_lr_experiments(
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_post",
        folder_name=folder_name,
        learning_rate=1e-3,
        norm_placement="post",
    ),  # gen_experim returns a list of experiment dictionaries
    [10**-3.25, 10**-2.75],
)


# GRAND_VARIATION_EXPERIMENTS = (
#     create_multi_seed_experiments(LR_SCHEDULE_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(INITIALIZATION_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(NORMALIZATION_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(POSITIONAL_ENCODING_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(ACTIVATION_EXPERIMENTS, SEEDS)
# )
GRAND_VARIATION_EXPERIMENTS = HISTORICAL_ORDER_LR_TUNE
