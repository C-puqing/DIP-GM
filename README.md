# Experiment Report

In this report, I want to conclude the first stage of the experiment. The report consists of experiment motivation, datasets, runtime, and hyper-parameters where I used in the experiment.

## Motivation and goal

The experiment has only one purpose, which is to investigate the influence of the exact graph matching method on the accuracy of the prediction on deep graph matching. Intuitively, it will definitely improve the matching performance. Here are some questions to be answered.

Question 1: Do exact methods matter when dealing with deep graph matching ?

Question 2: What is the drawback of exact methods compared with heuristic methods ?

## Datasets

I did this experiment on the standard datasets for keypoint matching Pascal VOC with Berkeley annotations and Willow ObjectClass. In addition, I follow the harder setup for Pascal VOC which proposed in [1] that avoids keypoint filtering as a prepocessing step.

## Runtime

All experiments were run on a single RTX 2080Ti GPU, Ubuntu 20.04 LTS system, Intel Xeon Silver 4216 CPU. The MIP solver implemented in here, was by `gurobipy` and each matching task has $|V_1||V_2| + |E_1||E_2|$ variables and $|V_1|+|V_2|+2|V_2|\cdot |E_1|$ constraints. Around one image pair spent two seconds.

## Hyperparameters

For fairness to evaluate the performance of exact methods, I didn't modify the hyperparameters used in [1]. The $\lambda$ is set to 80.0. The optimizer in use still is Adam with an initial learning rate of $2\times 10^{-3}$. Each batch process 8 image pairs. I seted the parameter of MIP solver "NumericFocus" to 2 and "Presolve" to 2.

## Results

Considered the expensive experiment time, I reduced the train iteration to 200 and test samples to 100. 

- **Pascal VOC with filitering keypoint as a preprocessing step.**

|    Class    | BB-GM | MILP  |
| :---------: | :---: | :---: |
|  aeroplane  | 44.64 | 43.12 |
|   bicycle   | 69.83 | 68.07 |
|    bird     | 72.43 | 72.95 |
|    boat     | 74.66 | 76.01 |
|   bottle    | 82.36 | 83.69 |
|     bus     | 91.87 | 88.94 |
|     car     | 75.86 | 75.43 |
|     cat     | 76.77 | 78.05 |
|    chair    | 41.51 | 39.00 |
|     cow     | 76.99 | 74.72 |
| diningtable | 69.87 | 89.33 |
|     dog     | 67.33 | 71.91 |
|    horse    | 74.97 | 75.99 |
|  motorbike  | 69.05 | 69.2  |
|   person    | 55.61 | 58.92 |
| pottedplant | 98.42 | 98.82 |
|    sheep    | 72.03 | 74.06 |
|    sofa     | 73.39 | 74.35 |
|    train    | 98.67 | 99.33 |
|  tvmonitor  | 94.46 | 96.68 |
|   AVERAGE   | 74.04 | 75.43 |

- **Pascal VOC with all keypoint**

|    Class    | BB-GM | MILP  |
| :---------: | :---: | :---: |
|  aeroplane  | 20.76 | 25.56 |
|   bicycle   | 67.85 | 67.04 |
|    bird     | 34.93 | 41.27 |
|    boat     | 39.16 | 35.79 |
|   bottle    | 88.98 | 89.90 |
|     bus     | 72.27 | 72.05 |
|     car     | 24.15 | 33.47 |
|     cat     | 53.41 | 59.49 |
|    chair    | 25.64 | 28.04 |
|     cow     | 49.95 | 59.83 |
| diningtable | 68.85 | 67.76 |
|     dog     | 46.84 | 52.80 |
|    horse    | 55.96 | 59.38 |
|  motorbike  | 51.75 | 58.92 |
|   person    | 23.72 | 27.60 |
| pottedplant | 95.62 | 97.01 |
|    sheep    | 41.76 | 50.08 |
|    sofa     | 37.24 | 40.91 |
|    train    | 86.17 | 86.17 |
|  tvmonitor  | 85.29 | 85.83 |
|   AVERAGE   | 53.51 | 56.94 |

- **Willow ObjectClass with no pre-train and finetune **

  |   Class    | BB-GM  |  MILP  |
  | :--------: | :----: | :----: |
  |    Car     | 97.40  | 95.00  |
  |    Duck    | 85.60  |  84.5  |
  |    Face    | 100.00 | 100.00 |
  | Motorbike  | 99.20  | 97.90  |
  | Winebottle | 94.60  | 94.90  |
  |  AVERAGE   | 95.36  | 94.46  |



[1] Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers
