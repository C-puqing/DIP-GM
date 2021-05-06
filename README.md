# Mix-ILP-and-Deep-Graph-Matching
This branch is the version of experiment 2.0. 
In this version, we use the novel loss fucntion instead of Hamming loss implemted by "DIFFERENTIATION OF BLACKBOX COMBINATORIAL
SOLVERS".

# The novel loss fucntion
$$
L( \textbf{v}, \textbf{t})=\frac{1}{|V_1|.|V_2| } \mathbf{1}_{|V_1|.|V_2|}^{T} . (\textbf{t}\odot(1-\textbf{v})+(1-\textbf{t})\odot\textbf{v})
$$

All experiment results are in the folder `results/.`, and the experiment configuration file is in the folder `experiments/`.
