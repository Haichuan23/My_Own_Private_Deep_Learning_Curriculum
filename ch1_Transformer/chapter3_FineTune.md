# Chapter 3: Fine-Tuning


# LoRA:

Low-Rank Adapatation: </br>
Instead of training the entire weight matrix, you inject (much) smaller trainable adapater matrices into specific layers (say key, query, value in transformer), and freeze the rest of the matrices. So instead of training W, we train A@B, where A and B have much smaller dimensions. The new matrix weight is `W' = W + ΔW = W + A @ B`. </br>

Quantization: </br>
Instead of using 32 bits to represent float, we use only 4 bits. This lowers precision, but reduces memory storage and speeds up inference. </br>

Implentation detail: LayerNorm's parameters are sensitive, so you should not freeze them during training and should be represented in their original precision (say float32). </br>

During training and inference, the low precision quantised weight is read from the storage, temporarily converted to their high precision translation, do the computation, and then discard the high precision translation. </br>


# RLHF Pipeline

## Step 1: Fine Tune a Reward model

1. Get pairwise comparion data (for example, from hugging face/chatbot_arena, the data should contain a prompt, two responses, and a label indicating who is the winner)
2. Filter the dataset, throw out prompts or responses that are too long (for memory and computation reasons)
3. We split the filtered pairwise comparison data into train set and validation set. We should sort the validation set so that longer sequences appear first.
[
For three reasons:</br>
a. You should not sort the training dataset because this will break the iid assumption of training data, but validation is not for training so we can sort it.</br>
b. The reason why you want to sort is you want samples with similar length to appear in the same batch. Recall that every sequence needs to be padded to the same length as the longest sequence in the batch. So suppose you have one long   sequence (say 1000 tokens), and one short sequence (say 100 tokens), then the short sequence needs 900 padding, which is wasteful.</br>
c. The largest sequence length in a batch determines the parameter for some matrices in the computation (say Q, K, V), so processing the larges sequence length first will make you encounter out of memory early on. You don't need to wait until the small ones that are already processed to encounter those errors.</br>
]
4. We need to tokenize our dataset (combining text and two separate responses together) using the tokenizer provided in the dataset Library (https://huggingface.co/docs/datasets/index). The tokenizer will return two things: (1) input_ids (the integer that the text is mapped to) (2) attention mask: tell you which token is text and which token is padding. We also need to have the labels indicating which model wins or loses.
5. Reward model is also a large transformer model, so it's computationally expensive and memory intensive to fine tune the reward model. Hence, we need to use LoRA and quantization (see BitsAndBytes). In the LoRa configurations, the three key parameters are:
[
  a. r is the rank of the LORA matrix, so suppose your original matrix W is $d^out \times d^in$, then your modification matrix B can be $d_out \times r$, and matrix A can be $r \times d^in$, and the modification is $B \times A$ </br>
  b. lora_alpha: This is a parameter controlling the strength of the LoRA update </br>
  c.  target_modules: This tells you which modules we would like to update, in the example, they change the weight to the q, k, v matrices in the attention mechanism </br>
]






