# Chapter 3: Supervised Fine Tuning


# LoRA:

Low-Rank Adapatation: </br>
Instead of training the entire weight matrix, you inject (much) smaller trainable adapater matrices into specific layers (say key, query, value in transformer), and freeze the rest of the matrices. So instead of training W, we train A@B, where A and B have much smaller dimensions. The new matrix weight is `W' = W + ΔW = W + A @ B`. </br>

Quantization: </br>
Instead of using 32 bits to represent float, we use only 4 bits. This lowers precision, but reduces memory storage and speeds up inference. </br>

Implentation detail: LayerNorm's parameters are sensitive, so you should not freeze them during training and should be represented in their original precision (say float32). </br>

During training and inference, the low precision quantised weight is read from the storage, temporarily converted to their high precision translation, do the computation, and then discard the high precision translation. </br>

# SFT Pipeline

## Some terminology refresh:
1. During training, training dataset is randomly shuffled and divided into batches. For each batch, the model will go through one forward and backward pass with parameter update. One epoch corresponds to the case that the model has seen through all data once.
2. Per ChatGPT, most large scale LM "pretraining" runs aren't descirbed in terms of epochs because the training size is so large and it's hard to see all texts once. Instead, people use total tokens seen as a metric. (GPT-2 (1.5B) is trained on roughly 52 billion tokens, LLaMA7B is trained on 1 trillion tokens, and GPT-3 (175B) is trained on roughly 300 billion tokens.)
3. In SFT training, for a 1B model and small dataset (< 10K examples), typically 5-10 epochs are enough. Bigger model has higher capacity, so less epochs are required to train them. If you train too many epochs on bigger models, then the risk of overfitting increases.


## Steps to SFT a LLM 
1. 

# RLHF Pipeline

## Step 1: Fine Tune a Reward model

1. Get pairwise comparion data (for example, from hugging face/chatbot_arena, the data should contain a prompt, two responses, and a label indicating who is the winner)
2. Filter the dataset, throw out prompts or responses that are too long (for memory and computation reasons)
3. We split the filtered pairwise comparison data into train set and validation set. We should sort the validation set so that longer sequences appear first.</br>
For three reasons:
a. You should not sort the training dataset because this will break the i.i.d. assumption of training data — but validation is not used for training, so we can sort it.

b. The reason why you want to sort is that you want samples with similar lengths to appear in the same batch.  
   - Recall that every sequence needs to be padded to the same length as the longest sequence in the batch.  
   - So suppose you have one long sequence (say 1000 tokens) and one short sequence (say 100 tokens), then the short one needs 900 tokens of padding — which is wasteful.

c. The largest sequence length in a batch determines the parameter size for some matrices in computation (e.g., Q, K, V).  
   - Processing the longest sequences first makes you encounter out-of-memory errors earlier.  
   - You don’t need to wait until the smaller ones are processed to discover those errors.

4. We need to tokenize our dataset (combining text and two separate responses together) using the tokenizer provided in the dataset Library (https://huggingface.co/docs/datasets/index). The tokenizer will return two things: (1) input_ids (the integer that the text is mapped to) (2) attention mask: tell you which token is text and which token is padding. We also need to have the labels indicating which model wins or loses.
5. Reward model is also a large transformer model, so it's computationally expensive and memory intensive to fine tune the reward model. Hence, we need to use LoRA and quantization (see BitsAndBytes). In the LoRa configurations, the three key parameters are: </br>
[
  a. r is the rank of the LORA matrix, so suppose your original matrix W is $d_{out} \times d_{in}$, then your modification matrix B can be $d_{out} \times r$, and matrix A can be $r \times d_{in}$, and the modification is $B \times A$ </br>
  b. lora_alpha: This is a parameter controlling the strength of the LoRA update </br>
  c.  target_modules: This tells you which modules we would like to update, in the example, they change the weight to the q, k, v matrices in the attention mechanism </br>
]
6. We then start to train the reward model. After downloading a public model, we will read the hidden state of the last non-padding token in each batch (retrieved using attention mask from tokenizer), and train a linear classifier that predicts which model wins. We use cross-entroypy loss to train the reward model. For the details, please see HuggingFace Trainer API, data collator, LoRa adaptor </br>
Some sample code for the final training pipeline: https://colab.research.google.com/drive/1B4NPpZzKLBxdk_zMqp4MM4026-g0wr3J?usp=sharing

## Step 2: The implementation of the PPO Algorithm



