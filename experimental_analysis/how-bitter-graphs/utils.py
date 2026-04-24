import numpy as np
import seaborn as sns
from constants import *

MAX_PRECISION, MAX_SIG_FIG = 2,3
STEPWISE = True


def rotary_multiplier(C):
    return np.full_like(C, LEARNED_TO_ROTARY_MULTIPLIER)

def pre_layer_norm_multiplier(C):
    return np.full_like(C, POST_TO_PRE_NORM_MULTIPLIER)

def sqrt_cosine_scheduler_multiplier(C):
    return np.full_like(C, SQRT_COSINE_SCHEDULER_MULTIPLIER)

def pre_rms_multiplier(C):
    return np.full_like(C, PRE_TO_RMS_MULTIPLIER)

def sinusoidal_learned_multiplier(C):
    return np.full_like(C, SINUSOIDAL_TO_LEARNED_MULTIPLIER)

def gelu_swiglu_multiplier(C):
    return np.full_like(C, GELU_TO_SWIGLU_MULTIPLIER)

def mixture_of_experts_multiplier(C):
    return np.full_like(C, MIXTURE_OF_EXPERTS_MULTIPLIER)

def transformer_2017_scaling(C):
    return A_T17* C**(alpha_T17)

def inverse_transformer_2017_scaling(L_adj):
    return (L_adj/A_T17) ** (1/alpha_T17)



def transformer_optimal_scaling(C):
    return A_TM* C**(alpha_TM)

def inverse_transformer_optimal_scaling(L_adj):
    return (L_adj/A_TM) ** (1/alpha_TM)


def lstm_optimal_scaling(C):
    return A_L * C ** (alpha_L)

def lstm_inverse_optimal_scaling(L_adj):
    return (L_adj/A_L) ** (1/alpha_L)


def lstm_to_optimal_transformer_multiplier(C, kaplan_divide = False):
    lstm_to_transformer_multiplier = lstm_inverse_optimal_scaling(transformer_optimal_scaling(C))/C
    if kaplan_divide:
        return lstm_to_transformer_multiplier / chinchilla_rebalancing_multiplier(C)
    else:
        return lstm_to_transformer_multiplier

def lstm_to_2017_transformer_multiplier(C, kaplan_divide = False):
    lstm_to_transformer_m = lstm_inverse_optimal_scaling(transformer_2017_scaling(C))/C
    if kaplan_divide:
        return lstm_to_transformer_m / chinchilla_rebalancing_multiplier(C)
    else:
        return lstm_to_transformer_m

YEARLY_SCALING_MULTIPLIER = 4.1
# TODO not so sure this is exact enough
def ytc(year, y0=2013, C0=1e16, k=np.log(YEARLY_SCALING_MULTIPLIER)):
    # ytc = Year -> Compute   
    return C0 * np.exp(k * (year - y0))

def cty(C, y0=2013, C0=1e16, k=np.log(YEARLY_SCALING_MULTIPLIER)):
    # cty = Compute -> Year
    return y0 + np.log(C / C0) / k


def compute_ceg_functions(C, innovations=None, cumulative=False, stepwise=STEPWISE, years=None):
    if innovations is None:
        innovations = INNOVATIONS
    
    C = np.atleast_1d(C)
    
    if years is not None:
        years = np.atleast_1d(years)
        assert C.size == years.size
    
    ceg_functions = {}
    for innovation_key, innovation_config in innovations.items():
        if stepwise:
            if years is not None:
                ceg_functions[innovation_key] = np.where(years >= innovation_config['year'], innovation_config['func'](C), 1)
            else:
                ceg_functions[innovation_key] = np.where(C >= ytc(innovation_config['year']), innovation_config['func'](C), 1)
        else:
            ceg_functions[innovation_key] = innovation_config['func'](C)
    
    if not cumulative:
        return ceg_functions
    
    cumulative_ceg_functions = {}
    product_so_far = np.ones_like(ceg_functions[next(iter(innovations))])
    for key in innovations:
        product_so_far = product_so_far * ceg_functions[key]
        cumulative_ceg_functions[key] = product_so_far.copy()
    
    return cumulative_ceg_functions

#############################################

def compute_ceg_statistics(growth_year_min=2014, growth_year_max=2023,
                           innovations=None):
    if innovations is None:
        innovations = INNOVATIONS
    
    
    C_min = np.floor(ytc(growth_year_min))
    C_max = np.ceil(ytc(growth_year_max))
    C = np.array(sorted(list(np.logspace(np.log10(C_min), np.log10(C_max), 1000))
                         + [C_min, C_max]))
    
    growth_year_min_idx = np.argmin(np.abs(C - ytc(growth_year_min)))
    growth_year_max_idx = np.argmin(np.abs(C - ytc(growth_year_max)))
    
    ceg_functions = compute_ceg_functions(C, innovations=innovations)
    cumulative_ceg_functions = compute_ceg_functions(C, innovations=innovations, cumulative=True)
    
    oldest_innovation, newest_innovation = list(innovations)[0], list(innovations)[-1]
    total_cumulative_growth = cumulative_ceg_functions[newest_innovation][growth_year_max_idx]
    initial_cumulative_growth = cumulative_ceg_functions[oldest_innovation][growth_year_min_idx]
    scale_dependent_growth = ceg_functions['chinchilla'][growth_year_max_idx] * ceg_functions['kaplan'][growth_year_max_idx]
    scale_invariant_growth = total_cumulative_growth / scale_dependent_growth
    
    return {
        'total_cumulative_growth': total_cumulative_growth,
        'initial_cumulative_growth': initial_cumulative_growth,
        'scale_dependent_growth': scale_dependent_growth,
        'scale_invariant_growth': scale_invariant_growth,
        'growth_year_min': growth_year_min,
        'growth_year_max': growth_year_max,
        'total_transformer_growth': ceg_functions['kaplan'][growth_year_max_idx],
        'total_chinchilla_growth': ceg_functions['chinchilla'][growth_year_max_idx]
    }



def format_sigfigs(num, sigfigs = MAX_SIG_FIG, max_precision = MAX_PRECISION):
    from math import log10, floor

    if num == 0:
        return "0"

    magnitude = floor(log10(abs(num)))
    rounded = round(num, -magnitude + sigfigs - 1)

    if magnitude >= sigfigs - 1:
        decimal_places_for_sigfigs = 0
    else:
        decimal_places_for_sigfigs = sigfigs - magnitude - 1

    decimal_places = min(decimal_places_for_sigfigs, max_precision)
    rounded = round(rounded, decimal_places)
    formatted = f"{rounded:,.{decimal_places}f}"

    return formatted

# More tough test cases
assert format_sigfigs(999.9, 2, 0) == "1,000"        # Rounds up to change magnitude
assert format_sigfigs(0.9999, 2, 1) == "1.0"         # Small number rounding up to 1
assert format_sigfigs(-12345, 3, 2) == "-12,300"     # Negative number
assert format_sigfigs(1234567890, 4, 0) == "1,235,000,000"  # Very large number
assert format_sigfigs(0.000456, 2, 5) == "0.00046"   # Very small number
assert format_sigfigs(51.6, 2, 0) == "52"            # Rounding at boundary
assert format_sigfigs(1.995, 2, 1) == "2.0"          # Precision limit causes rounding up
assert format_sigfigs(99999, 3, 1) == "100,000"      # Rounds up to 100k
assert format_sigfigs(0.0999, 1, 2) == "0.10"        # 1 sig fig on small number
assert format_sigfigs(7654321, 5, 3) == "7,654,300"  # 5 sig figs with restrictive precision
assert format_sigfigs(450.5, 3, 1) == "450"  




INNOVATIONS = dict(sorted({
    'kaplan': {
        'func': lambda C: lstm_to_2017_transformer_multiplier(C, kaplan_divide=True),
        'year': 2017,
        'label': 'Kaplan Transformer',
        'tiny': False
    },
    'sinusoidal': {
        'func': sinusoidal_learned_multiplier,
        'year': 2018,
        'label': 'Sinusoidal→Learned',
        'tiny': True
    },
    'sqrt_cosine': {
        'func': sqrt_cosine_scheduler_multiplier,
        'year': 2018,
        'label': 'Sqrt→Cosine Scheduler',
        'tiny': True
    },
    'pre_layer': {
        'func': pre_layer_norm_multiplier,
        'year': 2019,
        'label': 'Post→Pre Layer Norm',
        'tiny': False
    },
    'pre_rms': {
        'func': pre_rms_multiplier,
        'year': 2020,
        'label': 'Pre→RMS',
        'tiny': True
    },
    'rotary': {
        'func': rotary_multiplier,
        'year': 2021,
        'label': 'Learned→Rotary',
        'tiny': True
    },
    'moe': {
        'func': mixture_of_experts_multiplier,
        'year': 2021,
        'label': 'Mixture of Experts',
        'tiny': False
    },
    'chinchilla': {
        'func': chinchilla_rebalancing_multiplier,
        'year': 2022,
        'label': 'Chinchilla Rebalancing',
        'tiny': False
    }
}.items(), key=lambda x: x[1]['year']))


colors = list(sns.color_palette('tab20', 8))
for innovation, color in zip(INNOVATIONS, colors):
    INNOVATIONS[innovation]['color'] = color

for innovation, color in zip(INNOVATIONS, colors):
    INNOVATIONS[innovation]['color'] = color

