# Chapter 3: Fine-Tuning






## LoRA:

Low-Rank Adapatation: </br>
Instead of training the entire weight matrix, you inject (much) smaller trainable adapater matrices into specific layers (say key, query, value in transformer), and freeze the rest of the matrices. So instead of training W, we train A@B, where A and B have much smaller dimensions. The new matrix weight is `W' = W + ΔW = W + A @ B`. </br>

Quantization: </br>
Instead of using 32 bits to represent float, we use only 4 bits. This lowers precision, but reduces memory storage and speeds up inference. </br>

Implentation detail: LayerNorm's parameters are sensitive, so you should not freeze them during training and should be represented in their original precision (say float32). </br>

During training and inference, the low precision quantised weight is read from the storage, temporarily converted to their high precision translation, do the computation, and then discard the high precision translation. </br>
