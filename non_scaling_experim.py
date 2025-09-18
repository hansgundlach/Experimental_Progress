from experiment_utils import (
    create_multi_seed_experiments,
    create_multi_lr_experiments,
    calculate_transformer_params,
    gen_experim,
    get_base_config,
)

# activation variation experiments
ACTIVATION_EXPERIMENTS = (
    gen_experim(
        64, label="64d_activation_experiment", learning_rate=0.01, activation="gelu"
    )
    + gen_experim(
        64, label="64d_activation_experiment", learning_rate=0.01, activation="relu"
    )
    + gen_experim(
        64, label="64d_activation_experiment", learning_rate=0.01, activation="swiglu"
    )
)

# optimizer variation experiments ?
# positional encoding experiments
# norm experiments
# lr_schedule expeiremtns
# initalizaiton experiments
INITIALIZATION_EXPERIMENTS = (
    gen_experim(
        64,
        label="64d_initialization_experiment",
        learning_rate=0.01,
        init_scheme="xavier_normal",
    )
    + gen_experim(
        64,
        label="64d_initialization_experiment",
        learning_rate=0.01,
        init_scheme="kaiming_normal",
    )
    + gen_experim(
        64,
        label="64d_initialization_experiment",
        learning_rate=0.01,
        init_scheme="transformer",
    )
)


#normalization experiments 
NORMALIZATION_EXPERIMENTS = (
    gen_experim(
        64,
        label="64d_normalization_experiment",
        learning_rate=0.01,
        norm_type="layer",
    )
    + gen_experim(
        64,
        label="64d_normalization_experiment",
        learning_rate=0.01,
        norm_type="rms",
    )
    + gen_experim(
        64,
        label="64d_normalization_experiment",
        learning_rate=0.01,
        norm_type="layer_norn",
    )
)