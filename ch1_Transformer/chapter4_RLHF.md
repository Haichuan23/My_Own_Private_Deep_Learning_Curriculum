# RLHF Pipeline

## Step 1: Fine-Tune a Reward Model
   📎 **Sample training pipeline code**:  
   [Google Colab Notebook](https://colab.research.google.com/drive/1B4NPpZzKLBxdk_zMqp4MM4026-g0wr3J?usp=sharing)
1. **Get pairwise comparison data**  
   For example, from Hugging Face or Chatbot Arena. The data should contain:
   - A prompt
   - Two responses
   - A label indicating which response is the winner

2. **Filter the dataset**  
   Discard prompts or responses that are too long (to reduce memory and computation cost).

3. **Split the data into training and validation sets**  
   Sort the **validation set** so that longer sequences appear first. This is beneficial for three reasons:

   a. **Training data should not be sorted**, since that would break the i.i.d. assumption.  
      But validation data is not used for training, so we *can* sort it.

   b. **Sorting helps group similar-length samples in a batch**, which improves padding efficiency:  
      - All sequences in a batch must be padded to the length of the longest one.  
      - For example, a 1000-token and a 100-token sequence would require 900 padding tokens — wasteful!

   c. **Sorting helps expose memory limits earlier**:  
      - The longest sequence determines the memory footprint of matrices (e.g., Q, K, V in attention).  
      - If long sequences come early, you'll hit out-of-memory (OOM) errors sooner instead of wasting time.

4. **Tokenize the dataset**  
   Combine the prompt and two responses using the tokenizer provided in the [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index).  
   The tokenizer returns:
   - `input_ids`: integer-encoded text
   - `attention_mask`: indicates which tokens are actual text vs. padding  
   Also retain the label indicating which model wins.

5. **Configure LoRA and quantization**  
   The reward model is a large transformer and expensive to train. Use **LoRA** and **quantization** (e.g., via [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)) to reduce memory and compute. Key LoRA parameters include:

   - **`r`**: the rank of the LoRA matrix  
   Suppose your original matrix W is $d_{out} \times d_{in}$, then your modification matrix B can be $d_{out} \times r$, and matrix A can be $r \times d_{in}$, and the modification is $B \times A$

   - **`lora_alpha`**: scaling factor controlling the strength of the LoRA update

   - **`target_modules`**: which parts of the model are updated  
     In the examoke, LoRA is applied to the **Q, K, V** matrices in the attention mechanism.

6. **Train the reward model**  
   After downloading a public model:
   - Use the tokenizer’s attention mask to find the hidden state of the **last non-padding token** in each sequence.
   - Feed this hidden state into a **linear classifier** to predict which model wins.
   - Use **cross-entropy loss** to train the model.

   For implementation details, see:
   - Hugging Face’s `Trainer` API
   - Data collator
   - LoRA adapter

## Step 2: The implementation of the PPO Algorithm
This is also the solution for Chapter 5 in deep learning curriculum. Chapter 5 in this notebook is devoted to the PPO algorithm. 



