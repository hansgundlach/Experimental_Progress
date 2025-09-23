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

# activation variation experiments
ACTIVATION_EXPERIMENTS = (
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_gelu",
        learning_rate=0.01,
        activation="gelu",
        folder_name="alg_mult",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_relu",
        learning_rate=0.01,
        activation="relu",
        folder_name="alg_mult",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_swiglu",
        learning_rate=0.01,
        activation="swiglu",
        folder_name="alg_mult",
    )
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
        label=f"{DIMENSION}d_transformer_init",
        learning_rate=0.01,
        init_scheme="transformer",
        folder_name="alg_mult",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_bert_gpt_init",
        learning_rate=0.01,
        init_scheme="bert_gpt",
        folder_name="alg_mult",
    )
)


# normalization experiments
NORMALIZATION_EXPERIMENTS = (
    # gen_experim(
    #     DIMENSION,
    #     label="64d_layer",
    #     folder_name="alg_mult",
    #     learning_rate=0.01,
    #     norm_type="layer",
    # )
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_rms",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        norm_type="rms",
    )
    # add no norm placement option
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_post",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        norm_type="layer",
        norm_placement="post",
    )
)

OPTIMIZER_EXPERIMENTS = (
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_adamw",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        optimizer="adamw",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_sgd",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        optimizer="sgd",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_adam",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        optimizer="adam",
    )
)


POSITIONAL_ENCODING_EXPERIMENTS = (
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_rotary",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        pos_encoding="rotary",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_sinusoidal",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        pos_encoding="sinusoidal",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_learned",
        folder_name="alg_mult",
        learning_rate=10 ** (-2.5),
        pos_encoding="learned",
    )
)

# you would probably have to do an lr tune for the linear warmup
LR_SCHEDULE_EXPERIMENTS = (
    gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_lr_cosine_warmup",
        folder_name="alg_mult",
        learning_rate=10 ** (-2),
        lr_schedule="cosine_warmup",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_linear_warmup",
        folder_name="alg_mult",
        learning_rate=0.01,
        lr_schedule="linear_warmup",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_lr_inverse_sqrt",
        folder_name="alg_mult",
        learning_rate=0.01,
        lr_schedule="inverse_sqrt",
    )
    + gen_experim(
        DIMENSION,
        label=f"{DIMENSION}d_lr_transformer",
        folder_name="alg_mult",
        learning_rate=10 ** (-2),
        lr_schedule="transformer",
    )
)

# GRAND_VARIATION_EXPERIMENTS = (
#     create_multi_lr_experiments(LR_SCHEDULE_EXPERIMENTS, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(INITIALIZATION_EXPERIMENTS, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(NORMALIZATION_EXPERIMENTS, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(POSITIONAL_ENCODING_EXPERIMENTS, NARROW_LR_SWEEP)
#     + create_multi_lr_experiments(ACTIVATION_EXPERIMENTS, NARROW_LR_SWEEP)
# )


# GRAND_VARIATION_EXPERIMENTS = (
#     LR_SCHEDULE_EXPERIMENTS
#     + INITIALIZATION_EXPERIMENTS
#     + NORMALIZATION_EXPERIMENTS
#     + POSITIONAL_ENCODING_EXPERIMENTS
#     + ACTIVATION_EXPERIMENTS
# )


# GRAND_VARIATION_EXPERIMENTS = (
#     create_multi_seed_experiments(LR_SCHEDULE_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(INITIALIZATION_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(NORMALIZATION_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(POSITIONAL_ENCODING_EXPERIMENTS, SEEDS)
#     + create_multi_seed_experiments(ACTIVATION_EXPERIMENTS, SEEDS)
# )

GRAND_VARIATION_EXPERIMENTS = LR_SCHEDULE_EXPERIMENTS
