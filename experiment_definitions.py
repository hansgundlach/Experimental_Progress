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
    label="16d_speed_experiment_new_token_limit",
    learning_rate=0.01,
    token_limit=8056480,
)

TRANSFORMER_LR_TUNE_MUP_STANDARD = create_multi_lr_experiments(
    gen_experim(
        32, label="32d_mup", learning_rate=0.01, use_mup=True, mup_base_width=32
    ),
    NARROW_LR_SWEEP,
)


TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR = EXPERIMENTS = (
    gen_experim(32, label="32d_test_experiment", learning_rate=0.01)
    + gen_experim(40, label="40d_test_experiment", learning_rate=0.01)
    + gen_experim(48, label="48d_test_experiment", learning_rate=0.01)
    + gen_experim(56, label="56d_test_experiment", learning_rate=0.01)
    + gen_experim(64, label="64d_test_experiment", learning_rate=0.01)
)


TRANSFORMER_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR = EXPERIMENTS = (
    gen_experim(32, label="32d_test_experiment", learning_rate=0.01, optimizer="sgd")
    + gen_experim(40, label="40d_test_experiment", learning_rate=0.01, optimizer="sgd")
    + gen_experim(48, label="48d_test_experiment", learning_rate=0.01, optimizer="sgd")
    + gen_experim(56, label="56d_test_experiment", learning_rate=0.01, optimizer="sgd")
    + gen_experim(64, label="64d_test_experiment", learning_rate=0.01, optimizer="sgd")
)


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
