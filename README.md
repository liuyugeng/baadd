# Backdoor Attacks Against Dataset Distillation

[![arXiv](https://img.shields.io/badge/arxiv-2301.01197-b31b1b)](https://arxiv.org/abs/2301.01197)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is official code of our NDSS 23 paper Backdoor Attacks Against Dataset Distillation.
Currently, we apply two distillation techniques, namely [Dataset Distillation (DD)](https://arxiv.org/pdf/1811.10959.pdf) and [Dataset Condensation with Gradient Matching (DC)](https://arxiv.org/pdf/2006.05929.pdf).
In the project, we propose two different backdoor attacks *NAIVEATTACK* and *DOORPING*.
NAIVEATTACK inserts a pre-defined trigger into the original training dataset before the distillation.
DOORPING is an advanced method, which optimizes the trigger during the distillation process.

