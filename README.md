# Mix-ILP-and-Deep-Graph-Matching (Unfinished)
This branch is the version of experiment 2.0. 
In this version, we use the novel loss fucntion instead of Hamming loss implemted by "DIFFERENTIATION OF BLACKBOX COMBINATORIAL
SOLVERS".

All experiment results are in the folder `results/.`, and the experiment configuration file is in the folder `experiments/`.

# The novel loss fucntion
$$
L(v, sv, t) = t \odot (1-v)+(1-t) \odot v + sv \odot (t-v)
$$
