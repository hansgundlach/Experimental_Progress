# Model Organism of Algorithmic Progress


<img width="1126" height="736" alt="image" src="https://github.com/user-attachments/assets/0f5c78c6-2d16-433e-bb65-ac3d3adb307e" />


Ultra efficient transformers for benchmarking the impact of algorithmic improvements. 


# Current Optionality
activation functions: relu, gelu, silu/swish, GLU, swigGLU
optimizers: sgd(heavy ball momentum), adam, adamW
learning rate schedule: none, warmup+cosine annealing, warmup+inverse square root
positional encodings: sinusoidal, rotary encodings
initializations: xavier normal, kaiming normal, transformer (layer dependent) initalization
regularization options: early stopping, gradient clipping

## Future Options
Label smoothing, ALiBi, MoE
RMSNorm, LayerNorm



## Optimizations:
- torch.compile
- autocast mixed precision 
- flash attention 
- multiprocessing experiments
- data loading optimization: keeping workers alive, pinning memory, prefetch factor


## Stack 
- pytorch 
- weights and biases
- MIT supercloud V100s

## Running guidlines 
run setup_dataset.py to setup wikitext and GTP2Tokenizer
sbatch main.sh
