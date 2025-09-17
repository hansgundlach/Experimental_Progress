# lstm_experiment_definitions.py
import math
import copy
from lstm_experiment_utils import (
    gen_lstm_experim,
    create_multi_seed_lstm_experiments,
    create_multi_lr_lstm_experiments,
    create_multi_lr_experiments,
    gen_lstm_experim,
    calculate_lstm_params,
    get_lstm_base_config,
)


TEST_EXPERIMENT = gen_lstm_experim(
    16, label="lstm_16d_test_experiment", learning_rate=0.01
)


NARROW_LR_SWEEP = [
    10 ** (-3),
    10 ** (-2.5),
    10 ** (-2),
    10 ** (-1.5),
    1e-1,
]


LSTM_LR_TUNE_MUP_STANDARD = create_multi_lr_experiments(
    gen_lstm_experim(
        32, label="32d_lstm_mup", learning_rate=0.01, use_mup=True, mup_base_width=32
    ),
    NARROW_LR_SWEEP,
)


LSTM_SCALING_EXPERIMENTS_OPTIMAL_LR = EXPERIMENTS = (
    gen_lstm_experim(
        32, label="32d_lstm_experiment", folder_name="lstm_scaling", learning_rate=0.01
    )
    # + gen_lstm_experim(
    #     48, label="48d_lstm_experiment", folder_name="lstm_scaling", learning_rate=0.01
    # )
    # + gen_lstm_experim(
    #     64, label="64d_lstm_experiment", folder_name="lstm_scaling", learning_rate=0.01
    # )
)

LSTM_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR = EXPERIMENTS = (
    gen_lstm_experim(
        32,
        label="32d_lstm_sgd_experiment",
        folder_name="lstm_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_lstm_experim(
        48,
        label="48d_lstm_sgd_experiment",
        folder_name="lstm_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
    + gen_lstm_experim(
        64,
        label="64d_lstm_sgd_experiment",
        folder_name="lstm_sgd_scaling",
        learning_rate=0.1,
        optimizer="sgd",
    )
)


LSTM_ALL_SCALE_LR_TUNE = create_multi_lr_experiments(
    LSTM_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
)


LSTM_SGD_ALL_SCALE_LR_TUNE = create_multi_lr_experiments(
    LSTM_SGD_SCALING_EXPERIMENTS_OPTIMAL_LR, NARROW_LR_SWEEP
)


# TBPTT comparison experiments
TBPTT_COMPARISON_EXPERIMENTS = (
    gen_lstm_experim(
        32,
        label="32d_lstm_tbptt_default",
        folder_name="tbptt_comparison",
        learning_rate=0.01,
        use_tbptt=True,
        tbptt_length=128,
        tbptt_stride=128,
    )
    + gen_lstm_experim(
        32,
        label="32d_lstm_tbptt_disabled",
        folder_name="tbptt_comparison",
        learning_rate=0.01,
        use_tbptt=False,
    )
    + gen_lstm_experim(
        32,
        label="32d_lstm_tbptt_64_length",
        folder_name="tbptt_comparison",
        learning_rate=0.01,
        use_tbptt=True,
        tbptt_length=64,
        tbptt_stride=64,
    )
)

# Diagnostic experiments to fix scaling plateau
LSTM_SCALING_DIAGNOSTIC = (
    # Test higher learning rates
    gen_lstm_experim(
        32,
        label="32d_lstm_lr_aggressive",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,  # 5x higher than current
        warmup_frac=0.05,  # 5x longer warmup
        use_tbptt=False,  # Disable TBPTT initially
    )
    + gen_lstm_experim(
        32,
        label="32d_lstm_lr_very_aggressive",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.03,  # 15x higher
        warmup_frac=0.1,  # 10x longer warmup
        use_tbptt=False,
    )
    # Test lower dropout
    + gen_lstm_experim(
        32,
        label="32d_lstm_low_dropout",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.05,  # Much lowwer
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.1,  # Lower output dropout
        warmup_frac=0.05,
        use_tbptt=False,
    )
    # Test longer sequences with TBPTT
    + gen_lstm_experim(
        32,
        label="32d_lstm_long_seq",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        sequence_length=256,  # 2x longer sequences
        tbptt_length=128,  # Half the sequence length
        warmup_frac=0.05,
        use_tbptt=True,
    )
    #
)


# test no dropu
TESTING_TBPTTT = (
    gen_lstm_experim(
        32,
        label="yy32d_lstm_no_dropout_002_warmup_tbpttfalse",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=False,
    )
    # Uncomment to add low dropout experiment
    # + gen_lstm_experim(
    #     32,
    #     label="32d_lstm_low_dropout_0.02_warmup",
    #     folder_name="lstm_scaling_diagnostic",
    #     learning_rate=0.01,
    #     input_dropout=0.05,   # Low input dropout
    #     hidden_dropout=0.0,   # No hidden dropout
    #     output_dropout=0.1,   # Low output dropout
    #     warmup_frac=0.02,
    #     use_tbptt=False,
    # )
    + gen_lstm_experim(
        32,
        label="yy32d_lstm_no_dropout_002_warmup_testtbptt64",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
    )
    + gen_lstm_experim(
        32,
        label="yy32d_lstm_no_dropout_002_warmup_testtbptt32",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
    )
)


LSTM_SCALING = (
    gen_lstm_experim(
        32,
        label="yy32d_lstm_scaling_bs64",
        folder_name="lstm_scaling",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        48,
        label="yy48d_lstm_scaling_bs64",
        folder_name="lstm_scaling",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        64,
        label="yy64d_lstm_scaling_bs64",
        folder_name="lstm_scaling",
        learning_rate=10 ** (-2.5),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
)

LSTM_SCALING_NO_STREAMING = (
    gen_lstm_experim(
        32,
        label="yy32d_lstm_scaling_bs64_no_stream",
        folder_name="lstm_scaling",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
    )
    + gen_lstm_experim(
        48,
        label="yy48d_lstm_scaling_bs64_no_stream",
        folder_name="lstm_scaling",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
    )
    + gen_lstm_experim(
        64,
        label="yy64d_lstm_scaling_bs64_no_stream",
        folder_name="lstm_scaling",
        learning_rate=10 ** (-2.5),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
    )
)


LSTM_SCALING_DIAGNOSTIC = gen_lstm_experim(
    48,
    label="32d_lstm_scaling_diagnostic",
    folder_name="lstm_scaling_diagnostic",
    learning_rate=0.01,
    input_dropout=0.0,  # No input dropout
    hidden_dropout=0.0,  # No hidden dropout
    output_dropout=0.0,  # No output dropout
    warmup_frac=0.02,
    use_tbptt=True,
    tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
    tbptt_stride=32,  # Match the length for non-overlapping windows
)


LSTM_VARIATIONS_SCALING = (
    gen_lstm_experim(
        48,
        label="48d_tbptt_dropoutll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.2,  # No input dropout
        hidden_dropout=0.1,  # No hidden dropout
        output_dropout=0.2,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
    )
    + gen_lstm_experim(
        48,
        label="48d_diff_seedll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        seed=456,
    )
    + gen_lstm_experim(
        48,
        label="48d_batchsize64ll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
    )
    + gen_lstm_experim(
        48,
        label="48d_tbptt32ll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=128,
    )
    + gen_lstm_experim(
        48,
        label="48d_tbptt32_batch64ll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=64,
    )
    + gen_lstm_experim(
        48,
        label="48d_tbptt32_batch64_dropoutll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.2,
        hidden_dropout=0.1,
        output_dropout=0.2,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=64,
    )
    + gen_lstm_experim(
        48,
        label="48d_tbptt32_batch64_strong_dropoutll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=0.01,
        input_dropout=0.6,
        hidden_dropout=0.5,
        output_dropout=0.6,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=64,
    )
)


#     gen_lstm_experim(
#         48,
#         label="32d_lstm_scaling_diagnostic",
#         folder_name="lstm_scaling_diagnostic",
#         learning_rate=0.01,
#         input_dropout=0.0,  # No input dropout
#         hidden_dropout=0.0,  # No hidden dropout
#         output_dropout=0.0,  # No output dropout
#         warmup_frac=0.02,
#         use_tbptt=True,
#         tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
#         tbptt_stride=32,  # Match the length for non-overlapping windows
#     )
#     + gen_lstm_experim(
#         48,
#         label="48d_lstm_scaling_diagnostic_256_seq",
#         folder_name="lstm_scaling_diagnostic",
#         learning_rate=0.01,
#         input_dropout=0.0,  # No input dropout
#         hidden_dropout=0.0,  # No hidden dropout
#         output_dropout=0.0,  # No output dropout
#         warmup_frac=0.02,
#         use_tbptt=True,
#         tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
#         tbptt_stride=64,  # Match the length for non-overlapping windows
#         sequence_length=256,
#     )

#     + gen_lstm_experim(
#         48,
#         label="48d_diagnostic64",
#         folder_name="lstm_scaling_diagnostic",
#         learning_rate=0.01,
#         input_dropout=0.0,  # No input dropout
#         hidden_dropout=0.0,  # No hidden dropout
#         output_dropout=0.0,  # No output dropout
#         warmup_frac=0.02,
#         use_tbptt=True,
#         tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
#         tbptt_stride=64,  # Match the length for non-overlapping windows
#     )
#     + gen_lstm_experim(
#         48,
#         label="48d_lstm_strongdropout",
#         folder_name="lstm_scaling_diagnostic",
#         learning_rate=0.01,
#         input_dropout=0.2,  # No input dropout
#         hidden_dropout=0.1,  # No hidden dropout
#         output_dropout=0.2,  # No output dropout
#         warmup_frac=0.02,
#         use_tbptt=True,
#         tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
#         tbptt_stride=64,  # Match the length for non-overlapping windows
#     )
# )


# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     LSTM_SCALING, NARROW_LR_SWEEP
# ) + gen_lstm_experim(
#     32,
#     label="yy32d_lstm_scaling_bs256lr001",
#     folder_name="lstm_scaling",
#     learning_rate=0.01,
#     input_dropout=0.0,  # No input dropout
#     hidden_dropout=0.0,  # No hidden dropout
#     output_dropout=0.0,  # No output dropout
#     warmup_frac=0.02,
#     use_tbptt=True,
#     tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
#     tbptt_stride=64,  # Match the length for non-overlapping windows
#     batch_size=256,
#     use_streaming=True,
#     streaming_reset_prob=0.01,
# )

LSTM_VARIATIONS_SCALING_2 = (
    gen_lstm_experim(
        64,
        label="64d_tbptt_dropoutll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2.5),
        input_dropout=0.2,  # No input dropout
        hidden_dropout=0.1,  # No hidden dropout
        output_dropout=0.2,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        64,
        label="64d_diff_seedll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2.5),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        seed=456,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        48,
        label="64d_batchsize64ll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2.5),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        batch_size=64,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        48,
        label="64d_tbptt32ll_lr001",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=128,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt32_batch64ll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2.5),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=64,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt32_batch64_dropoutll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2.5),
        input_dropout=0.2,
        hidden_dropout=0.1,
        output_dropout=0.2,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=64,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt32_batch64_strong_dropoutll",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2.5),
        input_dropout=0.6,
        hidden_dropout=0.5,
        output_dropout=0.6,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        batch_size=64,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt_nodrop_lr001",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
    )
)


LSTM_VARIATIONS_SCALING_3 = (
    gen_lstm_experim(
        64,
        label="melis_settings",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.6,  # No input dropout
        hidden_dropout=0.5,  # No hidden dropout
        output_dropout=0.7,  # No output dropout
        between_layers_dropout=0.3,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        batch_size=64,
        num_layers=1,
    )
    + gen_lstm_experim(
        64,
        label="melis_settings_low_dropout",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.2,  # No input dropout
        hidden_dropout=0.05,  # No hidden dropout
        output_dropout=0.2,  # No output dropout
        between_layers_dropout=0.3,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=32,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        batch_size=64,
        num_layers=1,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt_nodrop_lr001_4layers",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        num_layers=4,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt_nodrop_lr001_4layers",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        num_layers=4,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt_nodrop_lr001_2layers_005warmup",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.05,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt_nodrop_lr001_2layers_005warmup",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="64d_tbptt_nodrop_lr001_2layers_gradientclipping5",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10 ** (-2),
        input_dropout=0.0,  # No input dropout
        hidden_dropout=0.0,  # No hidden dropout
        output_dropout=0.0,  # No output dropout
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=64,  # Half the sequence length - this will make TBPTT actually work!
        tbptt_stride=64,  # Match the length for non-overlapping windows
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        num_layers=2,
        gradient_clipping=5.0,
    )
)


MELIS_SCALING = (
    # gen_lstm_experim(
    #     32,
    #     label="32melis_settings_low_dropout",
    #     folder_name="lstm_scaling_diagnostic",
    #     learning_rate=10**-2,
    #     input_dropout=0.2,
    #     hidden_dropout=0.05,
    #     output_dropout=0.2,
    #     between_layers_dropout=0.3,
    #     optimizer="adam",
    #     adam_beta1=0.0,
    #     adam_beta2=0.999,
    #     adam_epsilon=1e-9,
    #     weight_decay=1e-4,
    #     warmup_frac=0.02,
    #     use_tbptt=True,
    #     tbptt_length=32,
    #     tbptt_stride=32,
    #     tbptt_reset_hidden=False,
    #     use_streaming=True,
    #     streaming_reset_prob=0.01,
    #     effective_batch_size=64,
    #     num_layers=1,
    # )
    # + gen_lstm_experim(
    #     32,
    #     label="32melis_settings_low_dropout_128bs",
    #     folder_name="lstm_scaling_diagnostic",
    #     learning_rate=10**-2,
    #     input_dropout=0.2,
    #     hidden_dropout=0.05,
    #     output_dropout=0.2,
    #     between_layers_dropout=0.3,
    #     optimizer="adam",
    #     adam_beta1=0.0,
    #     adam_beta2=0.999,
    #     adam_epsilon=1e-9,
    #     weight_decay=1e-4,
    #     warmup_frac=0.02,
    #     use_tbptt=True,
    #     tbptt_length=32,
    #     tbptt_stride=32,
    #     tbptt_reset_hidden=False,
    #     use_streaming=True,
    #     streaming_reset_prob=0.01,
    #     effective_batch_size=128,
    #     num_layers=1,
    # )
    # + gen_lstm_experim(
    #     48,
    #     label="48melis_settings_low_dropout",
    #     folder_name="lstm_scaling_diagnostic",
    #     learning_rate=10**-2,
    #     input_dropout=0.2,
    #     hidden_dropout=0.05,
    #     output_dropout=0.2,
    #     between_layers_dropout=0.3,
    #     optimizer="adam",
    #     adam_beta1=0.0,
    #     adam_beta2=0.999,
    #     adam_epsilon=1e-9,
    #     weight_decay=1e-4,
    #     warmup_frac=0.02,
    #     use_tbptt=True,
    #     tbptt_length=32,
    #     tbptt_stride=32,
    #     tbptt_reset_hidden=False,
    #     use_streaming=True,
    #     streaming_reset_prob=0.01,
    #     effective_batch_size=64,
    #     num_layers=1,
    # )
    gen_lstm_experim(
        80,
        label="80melis_settings_low_dropout",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10**-1.5,
        input_dropout=0.2,
        hidden_dropout=0.05,
        output_dropout=0.2,
        between_layers_dropout=0.3,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        effective_batch_size=64,
        num_layers=1,
    )
    + gen_lstm_experim(
        96,
        label="96melis_settings_low_dropout",
        folder_name="lstm_scaling_diagnostic",
        learning_rate=10**-1.5,
        input_dropout=0.2,
        hidden_dropout=0.05,
        output_dropout=0.2,
        between_layers_dropout=0.3,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        warmup_frac=0.02,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        effective_batch_size=64,
        num_layers=1,
    )
)


GRAND_EXPERIMENT = MELIS_SCALING


#     gen_lstm_experim(
#         32,
#         label="32d_short_tbptt_length",
#         folder_name="lstm_scaling_diagnostic",
#         learning_rate=0.01,
#         input_dropout=0.0,  # No input dropout
#         hidden_dropout=0.0,  # No hidden dropout
#         output_dropout=0.0,  # No output dropout
#         warmup_frac=0.02,
#         use_tbptt=False,
#         tbptt_length=32,
#         tbptt_stride=32,
#     )

# )


# lr_tune_experiments standard
# LSTM_LR_TUNE_STANDARD = create_multi_lr_experiments(
#     gen_lstm_experim(
#         32, label="32d_standard", learning_rate=0.01, use_mup=True, mup_base_width=32
#     ),
#     NARROW_LR_SWEEP,
# )


# ========= Experiment definitions (customize labels & overrides below) =========
# TEST_EXPERIMENTS = [
#     {
#         "name": "lstm_september_testing",
#         "subexperiments": [
#             {
#                 "label": "lstm_with_tokens_estimate",
#                 "overrides": {"learning_rate": 0.001 * math.sqrt(4), "hidden_size": 16},
#             },
#         ],
#     },
# ]


# LSTM_OPTIMAL_SCALING = [
#     {
#         "name": "lstm_optimal_scaling",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                 },
#             },
#             {
#                 "label": "lstm_24d",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193e6 * 2,
#                     "hidden_size": 24,
#                 },
#             },
#             {
#                 "label": "lstm_32d",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 258e6 * 2,
#                     "hidden_size": 32,
#                 },
#             },
#             {
#                 "label": "lstm_48d",
#                 "overrides": {
#                     "learning_rate": 1e-2,
#                     "max_characters": 388e6 * 2,
#                     "hidden_size": 48,
#                 },
#             },
#             {
#                 "label": "lstm_64d",
#                 "overrides": {
#                     "learning_rate": 1e-2,
#                     "max_characters": 519e6 * 2,
#                     "hidden_size": 64,
#                 },
#             },
#         ],
#     },
# ]


# LSTM_SGD_OPTIMAL_SCALING = [
#     {
#         "name": "lstm_sgd_optimal_scaling",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_24d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 193.7e6 * 2,
#                     "hidden_size": 24,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_32d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 258.6e6 * 2,
#                     "hidden_size": 32,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_48d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 388.8e6 * 2,
#                     "hidden_size": 48,
#                     "optimizer": "sgd",
#                 },
#             },
#             {
#                 "label": "lstm_64d_sgd",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 519.9e6 * 2,
#                     "hidden_size": 64,
#                     "optimizer": "sgd",
#                 },
#             },
#         ],
#     },
# ]

# LSTM_SGD_MUP_SCALING = [
#     {
#         "name": "lsmt_sgd_mup_scaling",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_24d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 193.7e6 * 2,
#                     "hidden_size": 24,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_32d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 258.6e6 * 2,
#                     "hidden_size": 32,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_48d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 388.8e6 * 2,
#                     "hidden_size": 48,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "lstm_64d_sgd_mup",
#                 "overrides": {
#                     "learning_rate": 1e-1,
#                     "max_characters": 519.9e6 * 2,
#                     "hidden_size": 64,
#                     "optimizer": "sgd",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#         ],
#     },
# ]

# LSTM_VARIATIONS = [
#     {
#         "name": "lstm_variations",
#         "subexperiments": [
#             {
#                 "label": "lstm_24d_layernorm",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                 },
#             },
#             {
#                 "label": "lstm_24d_3_layers",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "use_layer_norm": True,
#                     "num_layers": 3,
#                 },
#             },
#             {
#                 "label": "lstm_24d_no_layer_norm",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "use_layer_norm": False,
#                 },
#             },
#             {
#                 "label": "lstm_24d_cosine_warmup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "lr_schedule": "cosine_warmup",
#                 },
#             },
#             {
#                 "label": "lstm_24d_inverse_sqrt",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "max_characters": 193.7e6,
#                     "seed": 123,
#                     "hidden_size": 24,
#                     "lr_schedule": "inverse_sqrt",
#                 },
#             },
#         ],
#     },
# ]


# LSTM_MUP_SCALING_EXPERIMENTS = [
#     {
#         "name": "muP_scaling_experiments",
#         "subexperiments": [
#             {
#                 "label": "lstm_16d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "hidden_size": 16,
#                     "max_characters": 129e6 * 2,
#                     "seed": 123,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Base width for muP scaling
#                 },
#             },
#             {
#                 "label": "lstm_24d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),
#                     "hidden_size": 24,
#                     "max_characters": 193.7e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Base width for muP scaling
#                 },
#             },
#             {
#                 "label": "lstm_32d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),  # Same base LR as 16d
#                     "hidden_size": 32,
#                     "max_characters": 258.6e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Same base width - muP should handle scaling
#                 },
#             },
#             {
#                 "label": "lstm_48d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),  # Same base LR as 16d
#                     "hidden_size": 48,
#                     "max_characters": 388.8e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Same base width - muP should handle scaling
#                 },
#             },
#             {
#                 "label": "lstm_64d_mup",
#                 "overrides": {
#                     "learning_rate": 10 ** (-1.5),  # Same base LR - muP handles scaling
#                     "hidden_size": 64,
#                     "max_characters": 519.9e6 * 2,
#                     "use_mup": True,
#                     "mup_base_width": 16,  # Same base width
#                 },
#             },
#         ],
#     },
# ]
