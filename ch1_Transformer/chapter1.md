# Chapter 1: Transformer

Prerequisite:
1. Any online RNN tutorial would work. <br>
Textbook see chapter 10 of deep learning:https://www.deeplearningbook.org/<br>

Reading Materials:
1. Attention Blog post <br>
   https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/<br>
   seq to seq and Attention implementation: https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html<br>
2. Attention is all you need: https://arxiv.org/pdf/1706.03762 <br>
（video analysis: https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.337.search-card.all.click&vd_source=75e16b30403690b6ad4ccdb9c2dbde46)
3. Illustrated Transformer Blog post <br>
    https://jalammar.github.io/illustrated-transformer/

Practical Implementation:
1. Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out. <br>
 https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6067s
 
# Solution to Deep Learning Curriculum Chapter 1:
Other people's solution can be seen at:
1. https://github.com/MatthewBaggins/deep_learning_curriculum/blob/master/1-Transformers.md
2. https://github.com/jpdonasolo/Deep-Learning-Curriculum/blob/main/transformers/transformers.ipynb
 
Implement a decoder-only transformer language model.

Here are some first principle questions to answer:
## Q1. 
What is different architecturally from the Transformer, vs a normal RNN, like an LSTM? (Specifically, how are recurrence and time managed?)
### Solution:
RNN does have an attention mechanism. In RNN, a token gets its understanding of the context solely by hidden state computation, but that information does not distinguish between other tokens in the context (aka, a token cannot attend to other tokens which are more relevant to itself).
We distinguish between encoder and decoder in the transformer. In the encoder, you can attend to the entire context. So for a token at t_th index, it can attend to tokens at place t+1, for example. However, in the decoder transformer, you can only attend to tokens before you (because you produce the token one by one, so you don’t have access to future tokens). This is more similar to RNN in terms of how time is managed. 

## Q2. 
Attention is defined as, Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. What are the dimensions for Q, K, and V? Why do we use this setup? What other combinations could we do with (Q,K) that also output weights?
### Solution:
Let B be the batch dimension, T be the context window length, C_1 be the embedding dimension of tokens (specified by your embedding algorithm), C_2 be the embedding dimension specified by the model (d_model / d_head)

Q: (B, T, C_2)
K: (B, T, C_2)
V: (B, T, C_2)

## Q3. 
Are the dense layers different at each multi-head attention block? Why or why not?
### Solution:
Clarification: Dense layer is the fully connected layer. Say for attention block i, the dense layer for Q should be W_Q_{i}. 
Their dimensionalities are the same, but their parameters should be different. Intuitively, each head will take different roles, so they will have different parameters. 


## Q4. 
Why do we have so many skip connections, especially connecting the input of an attention function to the output? Intuitively, what if we didn't?

### Solution
We are training deep neural networks, so skip connections are added to prevent gradients from becoming zero. If we don’t have a skip connection, then it’s possible that the model sees little improvement because the gradient is too small.


## Q5. Implementation

## Our implementation 
https://colab.research.google.com/drive/1VHvAcJD8FwZLncf8y7gU44zqbrRObAIg?usp=sharing, following Andrej Karpathy's Let's build GPT from scratch video (https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2299s) 



Now we'll actually implement the code. Make sure each of these is completely correct - it's very easy to get the small details wrong.

Implement the positional embedding function first.

Then implement the function which calculates attention, given (Q,K,V) as arguments.

Now implement the masking function.

Put it all together to form an entire attention block.

Finish the whole architecture.

If you get stuck, The Annotated Transformer may help, but don't just copy-paste the code.

To check you have the attention mask set up correctly, train your model on a toy task, such as reversing a random sequence of tokens. The model should be able to predict the second half of the sequence, but not the first.

Finally, train your model on the complete works of William Shakespeare. Tokenize the corpus by splitting at word boundaries (re.split(r"\b", ...)). Make sure you don't use overlapping sequences as this can lead to overfitting.


