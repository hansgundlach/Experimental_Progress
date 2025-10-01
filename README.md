# Model Organism of Algorithmic Progress

How much influence have new AI algorithms had on improveements in training efficiency? 

#Here We Examine a Few Important Improvements
activation functions: ReLU, GeLU, SwiGLU
optimizers: sgd(heavy ball momentum), adam, adamW
positional encodings: sinusoidal, learned, and rotary encodings
learning rate schedules: linear decay, cosine decay 
normalization: layernorm, rmsnorm 
architectures: LSTM, Transformer


## Stack 
- pytorch 
- weights and biases
- MIT supercloud V100s



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



## Running guidlines 
run setup_dataset.py to setup wikitext and GTP2Tokenizer
sbatch main.sh