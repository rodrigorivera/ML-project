# ML-project


This repo is for Skoltech ML course project which attempts to find the method producing the most accurate forecast for M5 Forecasting Competition.  Standard statistical and DeepLearning methods were developed to tackle this problem. The method of the most interest is GNN-RNN that was devoted to take into account hierar-chical structure of the data.

- Exploratory_analysis.ipynb: Data hierarchical structure analysis and justification of GNN architecture choice

- Transformer.ipynb: Transformer implementation adjusted to time series forecasting, requires cuda

- class_Graph.ipynb: Graph construction based on hierarchical groups (for further work with GNN)

- preprocessing.py: Data memory usage reduction

- theta_method.ipynb: Optimized Theta method implementation for time series forecasting

- very-fst-model-upd.ipynb: LightGBM approach and feature generation based on kaggle kernel

- preprocessing_lstm.py: Leading zeros deletion

- lstm.ipynb: LSTM implementation and training

- training_GCN.ipynb: Graph Convolution neural network implementation (in progress)







