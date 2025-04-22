#Model Organism of Algorithmic Progress

Ultra efficient transformers for benchmarking the impact of algorithmic improvements. 


# Optionality
activation functions: relu, gelu, silu/swish, GLU, swigGLU
optimizers: sgd(heavy ball momentum), adam, adamW
learning rate schedule: none, warmup+cosine annealing, warmup+inverse square root
positional encodings: sinusoidal, rotary encodings
initializations: xavier normal, kaiming normal, transformer (layer dependent) initalization




## Optimizations:
- torch.compile
- autocast mixed precision 
- flash attention 
- multiprocessing experiments
- data loading optimization: keeping workers alive, pinning memory, prefetch factor


##Stack 
- pytorch 
- weights and biases
- MIT supercloud V100s

