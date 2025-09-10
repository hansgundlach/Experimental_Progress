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

TEST_EXPERIMENT = gen_experim(16, label="16d_speed_experiment", learning_rate=0.01)

TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR = EXPERIMENTS = (
    gen_experim(32, label="32d_test_experiment", learning_rate=0.01)
    + gen_experim(40, label="40d_test_experiment", learning_rate=0.01)
    + gen_experim(48, label="48d_test_experiment", learning_rate=0.01)
    + gen_experim(56, label="56d_test_experiment", learning_rate=0.01)
    + gen_experim(64, label="64d_test_experiment", learning_rate=0.01)
    + gen_experim(72, label="72d_test_experiment", learning_rate=0.01)
    + gen_experim(80, label="80d_test_experiment", learning_rate=0.01)
    + gen_experim(88, label="88d_test_experiment", learning_rate=0.01)
    + gen_experim(96, label="96d_test_experiment", learning_rate=0.01)
    + gen_experim(104, label="104d_test_experiment", learning_rate=0.01)
    + gen_experim(112, label="112d_test_experiment", learning_rate=0.01)
    + gen_experim(120, label="120d_test_experiment", learning_rate=0.01)
    + gen_experim(128, label="128d_test_experiment", learning_rate=0.01)
)


# lr_tune_experiments standard
TRANSFORMER_LR_TUNE_STANDARD = create_multi_lr_experiments(
    gen_experim(
        32, label="32d_standard", learning_rate=0.01, use_mup=True, mup_base_width=32
    ),
    NARROW_LR_SWEEP,
)
