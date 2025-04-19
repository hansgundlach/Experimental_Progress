# Experimental_Progress

Replication of transformer scaling laws, with and without algorithmic improvement. 
Using 2 NIVIDIA V100 GPUs. 


## Optimizations:
- torch.compile
- autocast mixed precision 
- flash attention 
- multiprocessing experiments
- data loading optimization: keeping workers alive, pinning memory, prefetch factor


##Stack 
- pytorch 
= weights and biases

