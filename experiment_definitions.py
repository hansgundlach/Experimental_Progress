# Activation Function Experiments
ACTIVATION_EXPERIMENTS = [
    {
        "name": "Activation_Functions_Comparison",
        "subexperiments": [
            {"label": "GELU", "overrides": {"activation": "gelu"}},
            {"label": "ReLU", "overrides": {"activation": "relu"}},
            {"label": "SwiGLU", "overrides": {"activation": "swiglu"}},
        ],
    },
]

# OPTIMIZER_EXPERIMENTS = [
#     {
#         "name": "Optimizer_Experiments",
#         "subexperiments": [
#             {
#                 "label": "96d_adam_123",
#                 "overrides": {
#                     "hidden_dim": 96,
#                     "num_layers": 6,
#                     "num_heads": 6,
#                     "learning_rate": 0.0024,
#                     "wikitext_limit": 109764480 * 4,
#                     "seed": 123,
#                     "optimizer": "adam",
#                 },
#             },
#         ],
#     },
# ]


OPTIMIZER_EXPERIMENTS = [
    {
        "name": "Optimizer_Experiments",
        "subexperiments": [
            {
                "label": "96d_adam_123",
                "overrides": {
                    "hidden_dim": 96,
                    "num_layers": 6,
                    "num_heads": 6,
                    "learning_rate": 0.0024,
                    "wikitext_limit": 109764480 * 4,
                    "seed": 123,
                    "optimizer": "adam",
                },
            },
        ],
    },
]


LR_EXPERIMENTS = [
    {
        "name": "Learning_Rate_Determination",
        "subexperiments": [
            {
                "label": "0.001",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": 16205120 * 4,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "0.0001",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.0001,
                    "wikitext_limit": 16205120 * 4,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "0.00001",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.00001,
                    "wikitext_limit": 16205120 * 4,
                    "optimizer": "sgd",
                },
            },
        ],
    },
]

# Hidden Dimension Scaling Experiments (simple override approach)
HIDDEN_DIM_EXPERIMENTS = [
    {
        "name": "Hidden_Dim_Scaling",
        "subexperiments": [
            {
                "label": "16d",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": 16205120 * 4,
                },
            },
            {
                "label": "32d",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": 32901760 * 4,
                },
            },
            {
                "label": "64d",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                },
            },
            {
                "label": "96d",
                "overrides": {
                    "hidden_dim": 96,
                    "num_layers": 6,
                    "num_heads": 6,
                    "learning_rate": 0.0024,
                    "wikitext_limit": 109764480 * 4,
                },
            },
        ],
    },
]
HIDDEN_DIM_EXPERIMENTS_NO_ROTARY = [
    {
        "name": "Hidden_Dim_Scaling_No_Rotary",
        "subexperiments": [
            {
                "label": "16d_no_rotary",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": 16205120 * 4,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "32d_no_rotary",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": 32901760 * 4,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "64d_no_rotary",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "pos_encoding": "sinusoidal",
                },
            },
            {
                "label": "96d_no_rotary",
                "overrides": {
                    "hidden_dim": 96,
                    "num_layers": 6,
                    "num_heads": 6,
                    "learning_rate": 0.0024,
                    "wikitext_limit": 109764480 * 4,
                    "pos_encoding": "sinusoidal",
                },
            },
        ],
    },
]

# Overrides:
# {'hidden_dim': 24, 'num_layers': 3, 'num_heads': 1, 'learning_rate': 0.001224744871391589, 'batch_size': 32, 'gradient_accumulation_steps': 8, 'train_tokens': 24538080
HIDDEN_DIM_EXPERIMENTS_123 = [
    {
        "name": "Hidden_Dim_Scaling",
        "subexperiments": [
            {
                "label": "16d_123",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": 16205120 * 4,
                    "seed": 123,
                },
            },
            {
                "label": "24d_123",
                "overrides": {
                    "hidden_dim": 24,
                    "num_layers": 3,
                    "num_heads": 1,
                    "learning_rate": 0.001224744871391589,
                    "wikitext_limit": 24538080 * 4,
                    "seed": 123,
                },
            },
            # {'hidden_dim': 48, 'num_layers': 4, 'num_heads': 3, 'learning_rate': 0.0017320508075688772, 'batch_size': 32, 'gradient_accumulation_steps': 8, 'train_tokens': 50458560}
            # 2522928 for  48 d model
            {
                "label": "32d_123",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": 32901760 * 4,
                    "seed": 123,
                },
            },
            {
                "label": "64d_123",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "seed": 123,
                },
            },
            {
                "label": "96d_123",
                "overrides": {
                    "hidden_dim": 96,
                    "num_layers": 6,
                    "num_heads": 6,
                    "learning_rate": 0.0024,
                    "wikitext_limit": 109764480 * 4,
                    "seed": 123,
                },
            },
        ],
    },
]
HIDDEN_DIM_EXPERIMENTS_NO_ROTARY_123 = [
    {
        "name": "Hidden_Dim_Scaling_No_Rotary_123",
        "subexperiments": [
            {
                "label": "16d_no_rotary_123",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": 16205120 * 4,
                    "pos_encoding": "sinusoidal",
                    "seed": 123,
                },
            },
            {
                "label": "24d_no_rotary_123",
                "overrides": {
                    "hidden_dim": 24,
                    "num_layers": 3,
                    "num_heads": 1,
                    "learning_rate": 0.001224744871391589,
                    "wikitext_limit": 24538080 * 4,
                    "pos_encoding": "sinusoidal",
                    "seed": 123,
                },
            },
            {
                "label": "32d_no_rotary_123",
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
            {
                "label": "64d_no_rotary_123",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "pos_encoding": "sinusoidal",
                    "seed": 123,
                },
            },
            {
                "label": "96d_no_rotary_123",
                "overrides": {
                    "hidden_dim": 96,
                    "num_layers": 6,
                    "num_heads": 6,
                    "learning_rate": 0.0024,
                    "wikitext_limit": 109764480 * 4,
                    "pos_encoding": "sinusoidal",
                    "seed": 123,
                },
            },
        ],
    },
]


# SGD
HIDDEN_DIM_EXPERIMENTS_123_SGD = [
    {
        "name": "Hidden_Dim_Scaling_SGD",
        "subexperiments": [
            {
                "label": "16d_123_sgd",
                "overrides": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_heads": 1,
                    "learning_rate": 0.001,
                    "wikitext_limit": 16205120 * 4,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "24d_123_sgd",
                "overrides": {
                    "hidden_dim": 24,
                    "num_layers": 3,
                    "num_heads": 1,
                    "learning_rate": 0.001224744871391589,
                    "wikitext_limit": 24538080 * 4,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            # {'hidden_dim': 48, 'num_layers': 4, 'num_heads': 3, 'learning_rate': 0.0017320508075688772, 'batch_size': 32, 'gradient_accumulation_steps': 8, 'train_tokens': 50458560}
            # 2522928 for  48 d model
            {
                "label": "32d_123_sgd",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": 32901760 * 4,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "48d_123_sgd",
                "overrides": {
                    "hidden_dim": 48,
                    "num_layers": 4,
                    "num_heads": 3,
                    "learning_rate": 0.0017320508075688772,
                    "wikitext_limit": 50458560 * 4,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "64d_123_sgd",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
        ],
    },
]

NORM_EXPERIMENTS = [
    {
        "name": "Norm_Experiments",
        "subexperiments": [
            {
                "label": "32d_pre_standard_123",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "norm_type": "layernorm",
                    "norm_placement": "pre",
                },
            },
            {
                "label": "32d_post_layernorm_123",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "norm_type": "layernorm",
                    "norm_placement": "post",
                },
            },
            {
                "label": "32d_rmsnorm_123",
                "overrides": {
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 2,
                    "learning_rate": 0.001,
                    "wikitext_limit": int(32901760 * 4),
                    "seed": 123,
                    "norm_type": "rmsnorm",
                },
            },
        ],
    },
]

LR_SCHEDULE_EXPERIMENTS = [
    {
        "name": "LR_Schedule_Experiments",
        "subexperiments": [
            {
                "label": "32d_cosine_standard_123",
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
                "label": "32d_cosine_warmup_123",
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
                "label": "32d_inverse_sqrt_123",
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
                "label": "32d_transfomer_123",
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
                "label": "32d_linear_warmup_123",
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
LR_SCHEDULE_EXPERIMENTS_LARGE = [
    {
        "name": "LR_Schedule_Experiments",
        "subexperiments": [
            {
                "label": "64d_cosine_standard_123",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "seed": 123,
                    "lr_schedule": "cosine",
                },
            },
            {
                "label": "64d_cosine_warmup_123",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "seed": 123,
                    "lr_schedule": "cosine_warmup",
                },
            },
            {
                "label": "64d_inverse_sqrt_123",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "seed": 123,
                    "lr_schedule": "inverse_sqrt",
                },
            },
            {
                "label": "64d_transformer_123",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "seed": 123,
                    "lr_schedule": "transformer",
                },
            },
            {
                "label": "64d_linear_warmup_123",
                "overrides": {
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "num_heads": 4,
                    "learning_rate": 0.002,
                    "wikitext_limit": 68261120 * 4,
                    "seed": 123,
                    "lr_schedule": "linear_warmup",
                },
            },
        ],
    }
]
