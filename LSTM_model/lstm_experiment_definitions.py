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

# =================================================== FUNDAMENTAL EXPERIMENTS ===================================================
LSTM_SCALING_STUDY_TRADITIONAL = (
    # gen_lstm_experim(
    #     32,
    #     label="32d",
    #     folder_name="lstm_layer1",
    #     learning_rate=5.6234e-2,
    #     token_to_param_ratio=40,
    # )
    # + gen_lstm_experim(
    #     48,
    #     label="48d",
    #     folder_name="lstm_layer1",
    #     learning_rate=4.0157e-2,
    #     token_to_param_ratio=40,
    # )
    gen_lstm_experim(
        64,
        label="64d_huge_data_160",
        folder_name="lstm_layer1",
        learning_rate=3.1623e-2,
        token_to_param_ratio=160,
    )
    # + gen_lstm_experim(
    #     80,
    #     label="80d",
    #     folder_name="lstm_layer1",
    #     learning_rate=2.627e-2,
    #     token_to_param_ratio=40,
    # )
    # + gen_lstm_experim(
    #     104,
    #     label="104d",
    #     folder_name="lstm_layer1",
    #     learning_rate=2.1130e-2,
    #     token_to_param_ratio=40,
    # )
    + gen_lstm_experim(
        128,
        label="128d_80_huge_data",
        folder_name="lstm_layer1",
        learning_rate=1.7783e-2,
        token_to_param_ratio=80,
    )
    # + gen_lstm_experim(
    #     160,
    #     label="160d",
    #     folder_name="lstm_layer1",
    #     learning_rate=1.4775e-2,
    #     token_to_param_ratio=40,
    # )
    # gen_lstm_experim(
    #     192,
    #     label="192d",
    #     folder_name="lstm_layer1",
    #     learning_rate=1.269e-2,
    #     token_to_param_ratio=40,
    # )
    # + gen_lstm_experim(
    #     224,
    #     label="224d",
    #     folder_name="lstm_layer1",
    #     learning_rate=1.1173e-2,
    #     token_to_param_ratio=40,
    # )
    # gen_lstm_experim(
    #     256,
    #     label="256d",
    #     folder_name="lstm_layer1",
    #     learning_rate=1.0e-2,
    #     token_to_param_ratio=40,
    # )
)


# MELSIS dropout vs no dropout
NO_DROPOUT_COMPARISON = gen_lstm_experim(
    64,
    label="64d_melis_dropout",
    folder_name="appendix_ablation_study",
    learning_rate=3.1623e-2,
    token_to_param_ratio=40,
    input_dropout=0.6,
    hidden_dropout=0.3,
    output_dropout=0.7,
    between_layers_dropout=0.0,
) + gen_lstm_experim(
    64,
    label="64d_standard_no_dropout",
    folder_name="appendix_ablation_study",
    learning_rate=3.1623e-2,
    token_to_param_ratio=40,
)
NO_DROPOUT_COMPARISON_LR = create_multi_lr_lstm_experiments(
    NO_DROPOUT_COMPARISON, [10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1]
)


# =================================================== LR SWEEPS ===================================================
NARROW_LR_SWEEP = [
    10 ** (-3),
    10 ** (-2.5),
    10 ** (-2),
    10 ** (-1.5),
    10 ** (-1),
]


LSTM_LR_SWEEP = [10 ** (-2), 10 ** (-1.5), 10 ** (-1)]

# LSTM_VARIATIONS = (
#     gen_lstm_experim(
#         32,
#         label="32melis_nodroplr-2bs64",
#         folder_name="lstm_scale_doctor",
#         learning_rate=10**-2,
#         input_dropout=0.0,
#         hidden_dropout=0.0,
#         output_dropout=0.0,
#         between_layers_dropout=0.0,
#         optimizer="adam",
#         adam_beta1=0.0,
#         adam_beta2=0.999,
#         adam_epsilon=1e-9,
#         weight_decay=1e-4,
#         warmup_frac=0.05,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         use_streaming=True,
#         streaming_reset_prob=0.01,
#         target_effective_batch_size=128,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         32,
#         label="32melis_lowdrop_lr-2bs64adamstand",
#         folder_name="lstm_scale_doctor",
#         learning_rate=10**-2,
#         input_dropout=0.2,
#         hidden_dropout=0.05,
#         output_dropout=0.2,
#         between_layers_dropout=0.3,
#         optimizer="adam",
#         adam_beta1=0.9,
#         adam_beta2=0.999,
#         adam_epsilon=1e-9,
#         weight_decay=1e-4,
#         warmup_frac=0.05,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         use_streaming=True,
#         streaming_reset_prob=0.01,
#         target_effective_batch_size=128,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         32,
#         label="32melis_lowdrop_lr-2bs16",
#         folder_name="lstm_scale_doctor",
#         learning_rate=10**-2,
#         input_dropout=0.2,
#         hidden_dropout=0.05,
#         output_dropout=0.2,
#         between_layers_dropout=0.3,
#         optimizer="adam",
#         adam_beta1=0.0,
#         adam_beta2=0.999,
#         adam_epsilon=1e-9,
#         weight_decay=1e-4,
#         warmup_frac=0.05,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         target_effective_batch_size=128,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         32,
#         label="32melis_lowdrop_lr15bs128",
#         folder_name="lstm_scale_doctor",
#         learning_rate=10 ** (-1.5),
#         input_dropout=0.2,
#         hidden_dropout=0.05,
#         output_dropout=0.2,
#         between_layers_dropout=0.3,
#         optimizer="adam",
#         adam_beta1=0.0,
#         adam_beta2=0.999,
#         adam_epsilon=1e-9,
#         weight_decay=1e-4,
#         warmup_frac=0.05,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         target_effective_batch_size=128,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         32,
#         label="32melis_lr2bs64",
#         folder_name="lstm_scale_doctor",
#         learning_rate=10**-2,
#         input_dropout=0.2,
#         hidden_dropout=0.05,
#         output_dropout=0.02,
#         between_layers_dropout=0.0,
#         optimizer="adam",
#         adam_beta1=0.0,
#         adam_beta2=0.999,
#         adam_epsilon=1e-9,
#         weight_decay=1e-4,
#         warmup_frac=0.05,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         use_streaming=True,
#         streaming_reset_prob=0.01,
#         target_effective_batch_size=128,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         32,
#         label="32melis_lr2bs64sgd",
#         folder_name="lstm_scale_doctor",
#         learning_rate=10**-2,
#         input_dropout=0.2,
#         hidden_dropout=0.05,
#         output_dropout=0.02,
#         between_layers_dropout=0.0,
#         optimizer="sgd",
#         weight_decay=1e-4,
#         warmup_frac=0.05,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         use_streaming=True,
#         streaming_reset_prob=0.01,
#         target_effective_batch_size=128,
#         num_layers=1,
#     )
#     + gen_lstm_experim(
#         32,
#         label="32melis_lr2bs64wd0_nodrop",
#         folder_name="lstm_scale_doctor",
#         learning_rate=10**-2,
#         input_dropout=0.0,
#         hidden_dropout=0.0,
#         output_dropout=0.0,
#         between_layers_dropout=0.0,
#         optimizer="adam",
#         adam_beta1=0.0,
#         adam_beta2=0.999,
#         adam_epsilon=1e-9,
#         weight_decay=0,
#         warmup_frac=0.05,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         use_streaming=True,
#         streaming_reset_prob=0.01,
#         target_effective_batch_size=128,
#         num_layers=1,
#     )
# )


# MELIS_SCALING = (
#     gen_lstm_experim(
#         32,
#         label="32melis_steam",
#         folder_name="lstm_scaling   _diagnostic",
#         learning_rate=10 ** -(1.5),
#         input_dropout=0.0,
#         hidden_dropout=0.0,
#         output_dropout=0.0,
#         between_layers_dropout=0.0,
#         optimizer="adam",
#         adam_beta1=0.0,
#         adam_beta2=0.999,
#         adam_epsilon=1e-9,
#         use_tbptt=True,
#         tbptt_length=32,
#         tbptt_stride=32,
#         tbptt_reset_hidden=False,
#         use_streaming=True,
#         streaming_reset_prob=0.01,
#         target_effective_batch_size=64,
#         num_layers=2,
#     )


MELIS_SCALING = (
    gen_lstm_experim(
        32,
        label="32melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(2),
        between_layers_dropout=0.0,
        optimizer="adam",
    )
    + gen_lstm_experim(
        48,
        label="48melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
        optimizer="adam",
    )
    + gen_lstm_experim(
        64,
        label="64melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
    )
    + gen_lstm_experim(
        80,
        label="80melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
        # Removed num_layers override - let it use base config value for fair comparison
    )
    + gen_lstm_experim(
        128,
        label="128melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
        # Removed num_layers override - let it use base config value for fair comparison
    )
    + gen_lstm_experim(
        160,
        label="160melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
    )
    + gen_lstm_experim(
        256,
        label="256melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
    )
)


SEC_LENGTH_CHANGE = (
    gen_lstm_experim(
        32,
        label="32melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(2),
        between_layers_dropout=0.0,
        optimizer="adam",
        sequence_length=32,
        tbptt_length=32,
    )
    + gen_lstm_experim(
        48,
        label="48melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
        optimizer="adam",
    )
    + gen_lstm_experim(
        64,
        label="64melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
    )
    + gen_lstm_experim(
        80,
        label="80melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
        # Removed num_layers override - let it use base config value for fair comparison
    )
    + gen_lstm_experim(
        128,
        label="128melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
        # Removed num_layers override - let it use base config value for fair comparison
    )
    + gen_lstm_experim(
        160,
        label="160melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
    )
    + gen_lstm_experim(
        256,
        label="256melis_stream",
        folder_name="lstm_scaling_study",
        learning_rate=10**-2,
    )
)

CORRECTED_MELIS_SCALING = (
    gen_lstm_experim(
        32,
        label="new32_correction_bs64",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(1.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="new64_correction_bs64",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(1.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        96,
        label="new96_correction_bs64",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(1.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        128,
        label="new128_correction_bs64",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(2),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        160,
        label="new160_correction_bs64",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(2),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        192,
        label="new192_correction_bs64",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(2),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        256,
        label="new256_correction_bs64",
        folder_name="lstm_scaling_study",
        learning_rate=10 ** -(2),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
)


SGD_TUNE_LR_SWEEP = [
    10 ** (-2),
    10 ** (-1.5),
    1e-1,
    10 ** (-0.5),
]

MELIS_SCALING_SGD = (
    gen_lstm_experim(
        32,
        label="32melis_steam_sgd",
        folder_name="lstm_sgd",
        learning_rate=10 ** (-0.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="sgd",
        sgd_momentum=0.0,
        weight_decay=0,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=32,
        num_layers=2,
        grad_clip=5.0,
    )
    + gen_lstm_experim(
        48,
        label="48melis_steam_sgd",
        folder_name="lstm_sgd",
        learning_rate=10 ** (-0.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="sgd",
        sgd_momentum=0.0,
        weight_decay=0,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=32,
        num_layers=2,
        grad_clip=5.0,
    )
    + gen_lstm_experim(
        64,
        label="64melis_steam_sgd",
        folder_name="lstm_sgd",
        learning_rate=10 ** (-0.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="sgd",
        sgd_momentum=0.0,
        weight_decay=0,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=32,
        num_layers=2,
        grad_clip=5.0,
    )
    + gen_lstm_experim(
        80,
        label="80melis_steam_sgd",
        folder_name="lstm_sgd",
        learning_rate=10 ** (-0.5),
        input_dropout=0.2,
        hidden_dropout=0.05,
        output_dropout=0.2,
        between_layers_dropout=0.0,
        optimizer="sgd",
        sgd_momentum=0.0,
        weight_decay=0,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=32,
        num_layers=2,
        grad_clip=5.0,
        # Removed num_layers override - let it use base config value for fair comparison
    )
    + gen_lstm_experim(
        128,
        label="128melis_stream_sgd",
        folder_name="lstm_sgd",
        learning_rate=10 ** (-0.5),
        input_dropout=0,
        hidden_dropout=0,
        output_dropout=0,
        between_layers_dropout=0.0,
        optimizer="sgd",
        sgd_momentum=0.0,
        weight_decay=0,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=32,
        num_layers=2,
        grad_clip=5.0,
        # Removed num_layers override - let it use base config value for fair comparison
    )
)


MELIS_DEBUG = gen_lstm_experim(
    96,
    label="96_correction_bs64",
    folder_name="lstm_scaling_study",
    learning_rate=10 ** -(1.5),
    input_dropout=0.0,
    hidden_dropout=0.0,
    output_dropout=0.0,
    between_layers_dropout=0.0,
    optimizer="adam",
    adam_beta1=0.0,
    adam_beta2=0.999,
    adam_epsilon=1e-9,
    weight_decay=1e-4,
    use_tbptt=True,
    tbptt_length=32,
    tbptt_stride=32,
    tbptt_reset_hidden=False,
    use_streaming=True,
    streaming_reset_prob=0.01,
    target_effective_batch_size=64,
    num_layers=2,
)

# numb layers scaling

LAYER_VARIATION = (
    gen_lstm_experim(
        64,
        label="64_layer_1_variation",
        folder_name="lstm_ablation_study",
        learning_rate=10 ** -(1.5),
        num_layers=1,
        max_tokens_training=int(129e6 / 8),
    )
    + gen_lstm_experim(
        64,
        label="64_layer_2_variation",
        folder_name="lstm_ablation_study",
        learning_rate=10 ** -(1.5),
        num_layers=2,
        max_tokens_training=int(129e6 / 8),
    )
    + gen_lstm_experim(
        256,
        label="256_layer_1_variation",
        folder_name="lstm_ablation_study",
        learning_rate=10 ** -(2),
        num_layers=1,
        max_tokens_training=int(129e6 / 8),
    )
    + gen_lstm_experim(
        256,
        label="256_layer_2_variation",
        folder_name="lstm_ablation_study",
        learning_rate=10 ** -(2),
        num_layers=2,
        max_tokens_training=int(129e6 / 8),
    )
)


# experiments to test
# appendix ablation study

# APPENDIX_ABALATION_STUDY =

APPENDIX_ABALATION_STUDY = (
    gen_lstm_experim(
        64,
        label="64_w_dropout",
        folder_name="lstm_ablation_study",
        learning_rate=10 ** -(1.5),
        input_dropout=0.6,
        hidden_dropout=0.3,
        output_dropout=0.7,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="64_w_adam",
        folder_name="lstm_ablation_study",
        learning_rate=10 ** -(1.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
    )
    + gen_lstm_experim(
        64,
        label="64_no_stream_eval",
        folder_name="lstm_ablation_study",
        learning_rate=10 ** -(1.5),
        input_dropout=0.0,
        hidden_dropout=0.0,
        output_dropout=0.0,
        between_layers_dropout=0.0,
        optimizer="adam",
        adam_beta1=0.0,
        adam_beta2=0.999,
        adam_epsilon=1e-9,
        weight_decay=1e-4,
        use_tbptt=True,
        tbptt_length=32,
        tbptt_stride=32,
        tbptt_reset_hidden=False,
        use_streaming=True,
        streaming_reset_prob=0.01,
        target_effective_batch_size=64,
        num_layers=2,
        eval_streaming_like_train=False,
    )
)

# LSTM_LR_STUDY

LSTM_LR_STUDY = (
    gen_lstm_experim(
        32,
        label="32_lr_study",
        folder_name="lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        64,
        label="64_lr_study",
        folder_name="lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        128,
        label="128_lr_study",
        folder_name="lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    # 160
    + gen_lstm_experim(
        160,
        label="160_lr_study",
        folder_name="lstm_lr",
        learning_rate=10 ** -(1.5),
    )
    + gen_lstm_experim(
        256,
        label="256_lr_study",
        folder_name="lstm_lr",
        learning_rate=10 ** -(1.5),
    )
)


# LSTM SCALING STUDY
# 32 48 64 80 104 128 160 256


LSTM_SEQ_LENGTH = (
    gen_lstm_experim(
        32,
        label="32d",
        folder_name="lstm_seq_length",
        learning_rate=5.6234e-2,
        token_to_param_ratio=40,
        sequence_length=32,
        tbptt_length=32,
    )
    # + gen_lstm_experim(
    #     48,
    #     label="48d",
    #     folder_name="lstm_seq_length",
    #     learning_rate=4.0157e-2,
    #     token_to_param_ratio=40,
    #     sequence_length=32,
    #     tbptt_length=32,
    # )
    + gen_lstm_experim(
        64,
        label="64d",
        folder_name="lstm_seq_length",
        learning_rate=3.1623e-2,
        token_to_param_ratio=40,
        sequence_length=32,
        tbptt_length=32,
    )
    # + gen_lstm_experim(
    #     80,
    #     label="80d",
    #     folder_name="lstm_seq_length",
    #     learning_rate=2.627e-2,
    #     token_to_param_ratio=40,
    #     sequence_length=32,
    #     tbptt_length=32,
    # )
    + gen_lstm_experim(
        96,
        label="96d",
        folder_name="lstm_seq_length",
        learning_rate=1.269e-2,
        token_to_param_ratio=40,
        sequence_length=32,
        tbptt_length=32,
    )
    # + gen_lstm_experim(
    #     104,
    #     label="104d",
    #     folder_name="lstm_seq_length",
    #     learning_rate=2.1130e-2,
    #     token_to_param_ratio=40,
    #     sequence_length=32,
    #     tbptt_length=32,
    # )
    + gen_lstm_experim(
        128,
        label="128d",
        folder_name="lstm_seq_length",
        learning_rate=1.7783e-2,
        token_to_param_ratio=40,
        sequence_length=32,
        tbptt_length=32,
    )
    + gen_lstm_experim(
        160,
        label="160d",
        folder_name="lstm_seq_length",
        learning_rate=1.4775e-2,
        token_to_param_ratio=40,
        sequence_length=32,
        tbptt_length=32,
    )
    + gen_lstm_experim(
        256,
        label="256d",
        folder_name="lstm_seq_length",
        learning_rate=1.0e-2,
        token_to_param_ratio=40,
        sequence_length=32,
        tbptt_length=32,
    )
)

# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     APPENDIX_ABALATION_STUDY, NARROW_LR_SWEEP
# )s
# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     gen_lstm_experim(
#         48,
#         label="48_lr_study",
#         folder_name="lstm_lr_study",
#         learning_rate=10**-1.5,
#     ),
#     [10**-1.75, 10**-2, 10**-2.25, 10**-2.5, 10**-2.75, 10**-3],
# )

# GRAND_EXPERIMENT = (
#     + gen_lstm_experim(
#         32,
#         label="32_layer_1",
#         folder_name="new_junk_folder",
#         learning_rate=10**-1.5,
#     )
#     + gen_lstm_experim(
#         32,
#         label="32_layer_2",
#         folder_name="new_junk_folder",
#         learning_rate=10**-1.5,
#         num_layers=2,
#     )
# )

# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     LSTM_LR_STUDY, [10**-1.5, 10**-1.75, 10**-2, 10**-2.25, 10**-2.5, 10**-2.75, 10**-3]
# )

# maybe this is the paper that recommended 0 momentum : https://proceedings.mlr.press/v37/jozefowicz15.pdf
SGD_EXPERIMENT = create_multi_lr_experiments(
    gen_lstm_experim(
        64,
        label="new32_sgd_0_momentum_nowd",
        folder_name="new_lstm_sgd",
        learning_rate=10**-1.5,
        optimizer="sgd",
        sgd_momentum=0.0,
        weight_decay=0.0,
    )
    + gen_lstm_experim(
        64,
        label="new32_sgd_09_momentum_nowd",
        folder_name="new_lstm_sgd",
        learning_rate=10**-1.5,
        optimizer="sgd",
        sgd_momentum=0.9,
        weight_decay=0.0,
    ),
    [
        10**-1,
        10**-1.5,
        10**-2,
        10**-2.5,
        10**-3,
        10**-3.5,
    ],
    generate_summary=True,
)


GRAND_EXPERIMENT = gen_lstm_experim(
    64,
    label="new32_sgd_0_momentum_nowd",
    folder_name="new_lstm_sgd",
    learning_rate=10**-1,
    optimizer="sgd",
    sgd_momentum=0.0,
    weight_decay=0.0,
) + gen_lstm_experim(
    64,
    label="new32_sgd_09_momentum_nowd",
    folder_name="new_lstm_sgd",
    learning_rate=10**-1,
    optimizer="sgd",
    sgd_momentum=0.9,
    weight_decay=0.0,
)
# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     SGD_EXPERIMENT, [10**-1, 10**-1.5, 10**-2, 10**-2.5, 10**-3, 10**-3.5]
# )


# GRAND_EXPERIMENT = gen_lstm_experim(
#     128,
#     label="128_32_batch_size",
#     folder_name="new_lstm_sgd",
#     learning_rate=10**-1,
#     optimizer="sgd",
#     sgd_momentum=0.9,
#     target_effective_batch_size=32,
# )


# GRAND_EXPERIMENT = (
#     create_multi_lr_experiments(
#         LSTM_SEQ_LENGTH,
#         [
#             10**-1,
#             10**-1.5,
#             10**-2,
#             10**-2.5,
#             10**-3,
#         ],
#     )
#     + LSTM_SEQ_LENGTH
# )
# GRAND_EXPERIMENT = create_multi_lr_experiments(
#     LSTM_SEQ_LENGTH,
#     [
#         10 ** (-1.25),
#         10 ** (-1.75),
#         10 ** (-2.25),
#         10 ** (-2.75),
#         10 ** (-3.25),
#     ],
# )

GRAND_EXPERIMENT = NO_DROPOUT_COMPARISON_LR

# 48 experimetns
