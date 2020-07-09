# MANE
Multi-View Collaborative Network Embedding https://arxiv.org/abs/2005.08189

Embedding learning algorithm with two versions: <strong>attention (MANE+, semi-supervised version) </strong> and <strong>without attention (MANE, unsupervised version)</strong> on a multi-view / multi-network dataset.

Three different datasets /tasks are available:
1) Link prediction: Binary class 
2) Link prediction: Multi-class, i.e., edges have labels -- relationship mining
3) Node classification

Example datasets and input formats are provided.

Usage: 
1) args_parser file of a chosen task should be modified for parameter settings or other choices before running the code.
2) main file of a chosen task should be run. (e.g., python main_Node_Classification_MANE.py)

Experiments performed on:
scikit-learn	0.19.1
scipy	1.2.0	
torch	0.4.1
Python 3.5

Compatible with both cuda and cpu devices, depending on the user choice through arg_parser file. Also compatible with python2 and python3.
