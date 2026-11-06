import numpy as np

from parameter_loader import load_parameters, parameter_path

PARAMETER_PATH = parameter_path()
PARAMETERS = load_parameters(PARAMETER_PATH)

_multipliers = PARAMETERS["innovation_multipliers"]
LEARNED_TO_ROTARY_MULTIPLIER = _multipliers["learned_to_rotary"]
POST_TO_PRE_NORM_MULTIPLIER = _multipliers["post_to_pre_norm"]
SQRT_COSINE_SCHEDULER_MULTIPLIER = _multipliers["sqrt_to_cosine_scheduler"]
PRE_TO_RMS_MULTIPLIER = _multipliers["pre_to_rms"]
SINUSOIDAL_TO_ROTARY = _multipliers["sinusoidal_to_rotary"]
SINUSOIDAL_TO_LEARNED_MULTIPLIER = SINUSOIDAL_TO_ROTARY / LEARNED_TO_ROTARY_MULTIPLIER
GELU_TO_SWIGLU_MULTIPLIER = _multipliers["gelu_to_swiglu"]
MIXTURE_OF_EXPERTS_MULTIPLIER = _multipliers["mixture_of_experts"]

_interaction = PARAMETERS["interaction_normalization"]
INTERACTION_TOTAL = _interaction["total"]

SENSITIVITY_DEVIATION = PARAMETERS.get("sensitivity", {}).get("deviation", 0.0)

_scaling_laws = PARAMETERS["scaling_laws"]
A_L, alpha_L = _scaling_laws["lstm"]["A"], _scaling_laws["lstm"]["alpha"]
A_T17, alpha_T17 = (
    _scaling_laws["transformer_2017"]["A"],
    _scaling_laws["transformer_2017"]["alpha"],
)
A_TM, alpha_TM = (
    _scaling_laws["modern_transformer"]["A"],
    _scaling_laws["modern_transformer"]["alpha"],
)


if SENSITIVITY_DEVIATION:
    alpha_L_ = ((alpha_TM - alpha_L) * (-SENSITIVITY_DEVIATION) + (alpha_L))
    print(f"Updating alpha_L from {alpha_L} to {alpha_L_} for sensitivity deviation of {SENSITIVITY_DEVIATION}.")
    alpha_L = alpha_L_

def chinchilla_rebalancing_multiplier(C):
    chinchilla = PARAMETERS["chinchilla_rebalancing"]
    return (
        chinchilla["coefficient"]
        * C
        * (
            chinchilla["term1_coefficient"] * C ** chinchilla["term1_exponent"]
            + chinchilla["term2_coefficient"] * C ** chinchilla["term2_exponent"]
        )
        ** chinchilla["outer_exponent"]
    )


TRANSFORMER_INNOVATIONS = [LEARNED_TO_ROTARY_MULTIPLIER, 
                            POST_TO_PRE_NORM_MULTIPLIER,
                            SQRT_COSINE_SCHEDULER_MULTIPLIER, 
                            PRE_TO_RMS_MULTIPLIER, 
                            SINUSOIDAL_TO_LEARNED_MULTIPLIER,
                            GELU_TO_SWIGLU_MULTIPLIER
                    ]
MULTIPLICATIVE_TRANSFORMER_PRODUCT = np.prod(TRANSFORMER_INNOVATIONS)



INTERACTION_NORMALIZATION = _interaction["enabled"]
if INTERACTION_NORMALIZATION:
    (LEARNED_TO_ROTARY_MULTIPLIER, 
        POST_TO_PRE_NORM_MULTIPLIER,
        SQRT_COSINE_SCHEDULER_MULTIPLIER, 
        PRE_TO_RMS_MULTIPLIER, 
        SINUSOIDAL_TO_LEARNED_MULTIPLIER,
        GELU_TO_SWIGLU_MULTIPLIER
    ) = [el ** (np.log(INTERACTION_TOTAL)/np.log(MULTIPLICATIVE_TRANSFORMER_PRODUCT)) 
        for el in (LEARNED_TO_ROTARY_MULTIPLIER, 
                    POST_TO_PRE_NORM_MULTIPLIER,
                    SQRT_COSINE_SCHEDULER_MULTIPLIER, 
                    PRE_TO_RMS_MULTIPLIER, 
                    SINUSOIDAL_TO_LEARNED_MULTIPLIER,
                    GELU_TO_SWIGLU_MULTIPLIER) 
    ]

    assert(np.isclose(INTERACTION_TOTAL, np.prod([LEARNED_TO_ROTARY_MULTIPLIER, 
                    POST_TO_PRE_NORM_MULTIPLIER,
                    SQRT_COSINE_SCHEDULER_MULTIPLIER, 
                    PRE_TO_RMS_MULTIPLIER, 
                    SINUSOIDAL_TO_LEARNED_MULTIPLIER,
                    GELU_TO_SWIGLU_MULTIPLIER])))

