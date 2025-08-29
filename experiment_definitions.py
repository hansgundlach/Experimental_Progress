# # Activation Function Experiments

BASIC_TEST_EXPERIMENT = [
    {
        "name": "Basic_Test_Experiment",
        "subexperiments": [
            {
                "label": "32d_cosine_standard_123",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(3290176),
                    "seed": 123,
                    "lr_schedule": "cosine",
                },
            },
        ],
    },
]

ACTIVATION_EXPERIMENTS = [
    {
        "name": "activation_functions",
        "subexperiments": [
            {
                "label": "32d_gelu",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "activation": "gelu",
                },
            },
            {
                "label": "32d_relu",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "activation": "relu",
                },
            },
            {
                "label": "32d_swiglu",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "activation": "swiglu",
                },
            },
        ],
    },
]

OPTIMIZER_EXPERIMENTS = [
    {
        "name": "optimizer_experiments",
        "subexperiments": [
            {
                "label": "32d_sgd",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 0.03162277,
                    "wikitext_limit": 32901760 * 4,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "32d_adam",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 0.03162277,
                    "wikitext_limit": 32901760 * 4,
                    "seed": 123,
                    "optimizer": "adam",
                },
            },
            {
                "label": "32d_adamw",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 0.03162277,
                    "wikitext_limit": 32901760 * 4,
                    "seed": 123,
                    "optimizer": "adamw",
                },
            },
            {
                "label": "32d_rmsprop",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 0.03162277,
                    "wikitext_limit": 32901760 * 4,
                    "seed": 123,
                    "optimizer": "adamw",
                },
            },
            {
                "label": "32d_adagrad",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 0.03162277,
                    "wikitext_limit": 32901760 * 4,
                    "seed": 123,
                    "optimizer": "adamw",
                },
            },
        ],
    },
]


POS_ENCODING_EXPERIMENTS = [
    {
        "name": "pos_encoding",
        "subexperiments": [
            {
                "label": "32d_learned",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": 32901760 * 4,
                    "pos_encoding": "learned",
                    "seed": 123,
                },
            },
            {
                "label": "32d_sinusoidal",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": 32901760 * 4,
                    "pos_encoding": "sinusoidal",
                    "seed": 123,
                },
            },
        ],
    },
]


NORM_EXPERIMENTS = [
    {
        "name": "norm_experiments",
        "subexperiments": [
            {
                "label": "32d_pre_standard",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "norm_type": "layernorm",
                    "norm_placement": "pre",
                },
            },
            {
                "label": "32d_post_layernorm_123",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "norm_type": "layernorm",
                    "norm_placement": "post",
                },
            },
            {
                "label": "32d_rmsnorm_123",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "norm_type": "rmsnorm",
                },
            },
        ],
    },
]


INITIALIZATION_EXPERIMENTS = [
    {
        "name": "initialization_experiments",
        "subexperiments": [
            {
                "label": "32d_xavier_uniform",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "init_type": "xavier",
                },
            },
            {
                "label": "32d_transformer_scaled",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "init_type": "transformer_scaled",
                },
            },
            {
                "label": "32d_transformer_scaled",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "init_type": "default",
                },
            },
        ],
    },
]


LR_SCHEDULE_EXPERIMENTS = [
    {
        "name": "lr_schedule",
        "subexperiments": [
            {
                "label": "32d_cosine_standard",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "lr_schedule": "cosine",
                },
            },
            {
                "label": "32d_cosine_warmup",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "lr_schedule": "cosine_warmup",
                },
            },
            {
                "label": "32d_inverse_sqrt",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "lr_schedule": "inverse_sqrt",
                },
            },
            {
                "label": "32d_transfomer",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "lr_schedule": "transformer",
                },
            },
            {
                "label": "32d_linear_warmup",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "lr_schedule": "linear_warmup",
                },
            },
        ],
    },
]


# SGD_VARIATION_EXPERIMENTS = [
#     {
#         "label": "32d_cosine_warmup_sgd",
#         "overrides": {
#             "hidden_dim": 32,
#             "num_layers": 3,
#             "num_heads": 2,
#             "learning_rate": 10 ** (-1.5),
#             "wikitext_limit": int(32901760 * 4),
#             "seed": 123,
#             "optimizer": "sgd",
#         },
#     },
#     {
#         "label": "32d_inverse_sqrt_sgd",
#         "overrides": {
#             "hidden_dim": 32,
#             "num_layers": 3,
#             "num_heads": 2,
#             "learning_rate": 10 ** (-1.5),
#             "wikitext_limit": int(32901760 * 4),
#             "seed": 123,
#             "optimizer": "sgd",
#         },
#     },
#     {
#         "label": "32d_linear_warmup_sgd",
#         "overrides": {
#             "hidden_dim": 32,
#             "num_layers": 3,
#             "num_heads": 2,
#             "learning_rate": 10 ** (-1.5),
#             "wikitext_limit": int(32901760 * 4),
#             "seed": 123,
#             "optimizer": "sgd",
#         },
#     },
# ]

# Transformer Variations Experiments:

SGD_SCHEDULE_VARIATION_EXPERIMENTS = [
    {
        "name": "sgd_schedule_experiments",
        "subexperiments": [
            {
                "label": "32d_cosine_warmup_sgd",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": int(32901760 * 4) / 8,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "32d_inverse_sqrt_sgd",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": int(32901760 * 4) / 8,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "32d_linear_warmup_sgd",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": int(32901760 * 4) / 8,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
        ],
    }
]

# head variation experiment
TRANSFORMER_VARIATION_EXPERIMENTS_HEAD = [
    {
        "name": "transformer_head_experiments",
        "subexperiments": [
            {
                "label": "32d_1_head",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": int(32901760 * 4) / 2,
                    "seed": 123,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "32d_2_head",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": int(32901760 * 4) / 2,
                    "seed": 123,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "32d_3_head",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": int(32901760 * 4) / 2,
                    "seed": 123,
                    "optimizer": "sgd",
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
        ],
    },
]


# NEW_HYPER_PARAM_EXPERIMENTS = [
#     {
#         "name": "New_Hyper_Param_Experiments",
#         "subexperiments": [
#             {
#                 "label": "32d_standard_hyper_params",
#                 "overrides": {
#                     "hidden_dim": 32,
#                     "num_layers": 3,
#                     "num_heads": 2,
#                     "learning_rate": 0.001,
#                     "wikitext_limit": 32901760 * 4,
#                 },
#             },
#             {
#                 "label": "32d_new_hyper_params",
#                 "overrides": {
#                     "hidden_dim": 32,
#                     "num_layers": 2,
#                     "num_heads": 2,
#                     "learning_rate": 0.0008485,
#                     "wikitext_limit": 32901760 * 4,
#                 },
#             },
#             {
#                 "label": "half_tokens",
#                 "overrides": {
#                     "hidden_dim": 32,
#                     "num_layers": 2,
#                     "num_heads": 2,
#                     "learning_rate": 0.0008485,
#                     "wikitext_limit": 32901760 * 4 / 2,
#                 },
#             },
#             {
#                 "label": "increase_strid",
#                 "overrides": {
#                     "hidden_dim": 32,
#                     "num_layers": 2,
#                     "num_heads": 2,
#                     "learning_rate": 0.0008485,
#                     "wikitext_limit": 32901760 * 4,
#                     "stride": 128,
#                 },
#             },
#         ],
#     }
# ]

# effect of two changes
TWO_CHANGES_EXPERIMENTS = [
    {
        "name": "Two_Changes_Experiments",
        "subexperiments": [
            {
                "label": "128d_new_hyper_params",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 32901760 * 4,
                    "optimizer": "sgd",
                    "pos_encoding": "sinusoidal",
                },
            },
        ],
    },
]


# Hidden Dimension Scaling Experiments (simple override approach)
TRANSFORMER_SCALING_EXPERIMENTS = [
    {
        "name": "transformer_standard_scaling",
        "subexperiments": [
            {
                "label": "16d_standard_mup",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 16205120 * 4,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "24d_standard_mup",
                "overrides": {
                    "hidden_dim": 24,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 24538080 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "32d_standard_mup",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 32901760 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "40d_standard_mup",
                "overrides": {
                    "hidden_dim": 40,
                    "num_layers": 3,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 41127200 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "48d_standard_mup",
                "overrides": {
                    "hidden_dim": 48,
                    "num_layers": 3,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 50458560 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "64d_standard_mup",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 68261120 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
        ],
    },
]


# NO_ROTARY_SCALING_EXPERIMENT = [
#     {
#         "name": "scaling_experiments_sinusoidal",
#         "subexperiments": [
#             {
#                 "label": "16d_no_rotary",
#                 "overrides": {
#                     "hidden_dim": 16,
#                     "num_layers": 2,
#                     "num_heads": 1,
#                     "learning_rate": 10 ** (-1.5),
#                     "wikitext_limit": 16205120 * 4 * 2,
#                     "pos_encoding": "sinusoidal",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "32d_no_rotary",
#                 "overrides": {
#                     "hidden_dim": 32,
#                     "num_layers": 3,
#                     "num_heads": 1,
#                     "learning_rate": 10 ** (-1.5),
#                     "wikitext_limit": 32901760 * 4 * 2,
#                     "pos_encoding": "sinusoidal",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#             {
#                 "label": "64d_no_rotary",
#                 "overrides": {
#                     "hidden_dim": 64,
#                     "num_layers": 4,
#                     "num_heads": 1,
#                     "learning_rate": 10 ** (-1.5),
#                     "wikitext_limit": 68261120 * 4 * 2,
#                     "pos_encoding": "sinusoidal",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },
#         ],
#     },
# ]

NO_ROTARY_SCALING_EXPERIMENTS = [
    {
        "name": "transformer_no_rotary_scaling",
        "subexperiments": [
            {
                "label": "16d_mup_no_rotary",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 16205120 * 4,
                    "use_mup": True,
                    "mup_base_width": 16,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "24d_mup_no_rotary",
                "overrides": {
                    "hidden_dim": 24,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 24538080 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "32d_mup_no_rotary",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 32901760 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "40d_mup_no_rotary",
                "overrides": {
                    "hidden_dim": 40,
                    "num_layers": 3,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 41127200 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "48d_mup_no_rotary",
                "overrides": {
                    "hidden_dim": 48,
                    "num_layers": 3,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 50458560 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "64d_mup_no_rotary",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 68261120 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                    "pos_encoding": "sinusoidal",
                },
            },
        ],
    },
]

TRANSFORMER_SGD_SCALING_EXPERIMENTS = [
    {
        "name": "transformer_sgd_scaling",
        "subexperiments": [
            {
                "label": "16d_sgd",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 16205120 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "24d_sgd",
                "overrides": {
                    "hidden_dim": 24,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 24538080 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "32d_sgd",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 32901760 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "40d_sgd",
                "overrides": {
                    "hidden_dim": 40,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 41127200 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "48d_sgd",
                "overrides": {
                    "hidden_dim": 48,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 50458560 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
            {
                "label": "64d_sgd",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 10 ** (-1.5),
                    "wikitext_limit": 68261120 * 4 * 2,
                    "use_mup": True,
                    "mup_base_width": 16,
                },
            },
        ],
    },
]


# {
#                 "label": "96d_no_rotary",
#                 "overrides": {
#                     "hidden_dim": 96,
#                     "num_layers": 6,
#                     "num_heads": 6,
#                     "learning_rate": 0.0024,
#                     "wikitext_limit": 109764480 * 4,
#                     "pos_encoding": "sinusoidal",
#                     "use_mup": True,
#                     "mup_base_width": 16,
#                 },
#             },


# Overrides:
# {'hidden_dim': 24, 'num_layers': 3, 'num_heads': 1, 'learning_rate': 0.001224744871391589, 'batch_size': 32, 'gradient_accumulation_steps': 8, 'train_tokens': 24538080


# SGD


# LR_SCHEDULE_EXPERIMENTS_LARGE = [
#     {
#         "name": "LR_Schedule_Experiments",
#         "subexperiments": [
#             {
#                 "label": "64d_cosine_standard_123",
#                 "overrides": {
#                     "hidden_dim": 64,
#                     "num_layers": 4,
#                     "num_heads": 4,
#                     "learning_rate": 0.002,
#                     "wikitext_limit": 68261120 * 4,
#                     "seed": 123,
#                     "lr_schedule": "cosine",
#                 },
#             },
#             {
#                 "label": "64d_cosine_warmup_123",
#                 "overrides": {
#                     "hidden_dim": 64,
#                     "num_layers": 4,
#                     "num_heads": 4,
#                     "learning_rate": 0.002,
#                     "wikitext_limit": 68261120 * 4,
#                     "seed": 123,
#                     "lr_schedule": "cosine_warmup",
#                 },
#             },
#             {
#                 "label": "64d_inverse_sqrt_123",
#                 "overrides": {
#                     "hidden_dim": 64,
#                     "num_layers": 4,
#                     "num_heads": 4,
#                     "learning_rate": 0.002,
#                     "wikitext_limit": 68261120 * 4,
#                     "seed": 123,
#                     "lr_schedule": "inverse_sqrt",
#                 },
#             },
#             {
#                 "label": "64d_transformer_123",
#                 "overrides": {
#                     "hidden_dim": 64,
#                     "num_layers": 4,
#                     "num_heads": 4,
#                     "learning_rate": 0.002,
#                     "wikitext_limit": 68261120 * 4,
#                     "seed": 123,
#                     "lr_schedule": "transformer",
#                 },
#             },
#             {
#                 "label": "64d_linear_warmup_123",
#                 "overrides": {
#                     "hidden_dim": 64,
#                     "num_layers": 4,
#                     "num_heads": 4,
#                     "learning_rate": 0.002,
#                     "wikitext_limit": 68261120 * 4,
#                     "seed": 123,
#                     "lr_schedule": "linear_warmup",
#                 },
#             },
#         ],
#     }
# ]
