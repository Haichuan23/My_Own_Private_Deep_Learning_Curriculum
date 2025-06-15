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
 
Implement a decoder-only transformer language model.

Here are some first principle questions to answer:
What is different architecturally from the Transformer, vs a normal RNN, like an LSTM? (Specifically, how are recurrence and time managed?)
Attention is defined as, Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. What are the dimensions for Q, K, and V? Why do we use this setup? What other combinations could we do with (Q,K) that also output weights?
Are the dense layers different at each multi-head attention block? Why or why not?
Why do we have so many skip connections, especially connecting the input of an attention function to the output? Intuitively, what if we didn't?
Now we'll actually implement the code. Make sure each of these is completely correct - it's very easy to get the small details wrong.
Implement the positional embedding function first.
Then implement the function which calculates attention, given (Q,K,V) as arguments.
Now implement the masking function.
Put it all together to form an entire attention block.
Finish the whole architecture.
If you get stuck, The Annotated Transformer may help, but don't just copy-paste the code.
To check you have the attention mask set up correctly, train your model on a toy task, such as reversing a random sequence of tokens. The model should be able to predict the second half of the sequence, but not the first.
Finally, train your model on the complete works of William Shakespeare. Tokenize the corpus by splitting at word boundaries (re.split(r"\b", ...)). Make sure you don't use overlapping sequences as this can lead to overfitting.


