# Modified LeakGAN for peptide sequences

Quite a few changes have been made from the original implementation;
* kernel numbers have been decreased and fewer kernels have been selected because i currently lack a supercomputer
* training epochs have been drastically reduced


It seems to work after the changes. \
The model underwent 80 epochs of adversarial training. Checkpoints of the model at the 0th and 80th epoch are available in the root folder (`checkpoint0.pth.tar` and `checkpoint80.pth.tar` respectively) 

## Files 
* `active.txt`: contains the (positive) peptide sequences in plaintext
* `corpus.npy`: contains the (positive) peptide sequences in tokens for the model to interpret
* `chars.pkl`: pickled list of characters, used for decoding the output of the GAN back into sequences
* `encode_peptides.py`: contains functions used to generate the two aforementioned files from `active.txt`. 


## Output Files
* `finalgenerated0.npy`: contains 256 generated sequences by the model after the **0th epoch**.
* `generated_sequences0.txt`: contains these sequences in "human-readable" format (characters as amino acids).


* `finalgenerated.npy`: contains 256 generated sequences by the model after the **80th epoch**.
* `generated_sequences.txt`: contains these sequences in "human-readable" format (characters as amino acids).

## Misc
* `gen_corpus.npy`: I don't remember what this was for