# Introduction
This is my notebook, in which i have described a way to make translator using PyTorch. 

Working with text is a popular branch in machine learning research. Here are the most powerful technologies and mathematics solutions. In this topic i am going to implement translator from english to russian. 

Ready? Go!

# Basics

I am going to use seq2seq model and the most common of this kind of models is encoder-decoder model, which use a `recurrent neural network (RNN)` to encode the source (input) sentence into a single vector. When we have a vector of the sentence, we can decode this vector and get the translation of the sentence. This is how this works:

![](./img_trans/seq2seq1.png)

(P.S. at the photo described german-english translation, but english-russian works the same way. Just another tokens)

The input(source) sentence passes through the encoder, collects hidden states and then goes into decoder model. The first initial hidden state of decoder model is the last hidden state of the encoder model. `<sos>` and `<eos>` tokens mean that the sentence starts and sentence ends. 

Lets say we have <img src="https://render.githubusercontent.com/render/math?math=$X = \\{x_1, x_2, ... x_T\\}$> 

$X = \\{x_1, x_2 ... x_T\\}$, where $x_1 = `<sos>`, x_2 = guten$ and etc. The initial hidden state, $h_0$, is usually either initialized to zeros or a learned parameter.

$X = \\{x_1, x_2, ..., x_T\\}$, where $x_1 = \\text{<sos>}, x_2 = \\text{guten}$, etc. The initial hidden state, $h_0$, is usually either initialized to zeros or a learned parameter.

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
