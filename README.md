# Introduction
This is my notebook, in which i have described a way to make translator using PyTorch. 

Working with text is a popular branch in machine learning research. Here are the most powerful technologies and mathematics solutions. In this topic i am going to implement translator from english to russian. 

Ready? Go!

# Basics

I am going to use seq2seq model and the most common of this kind of models is encoder-decoder model, which use a `recurrent neural network (RNN)` to encode the source (input) sentence into a single vector. When we have a vector of the sentence, we can decode this vector and get the translation of the sentence. This is how this works:

![](./img_trans/seq2seq1.png)

(P.S. at the photo described german-english translation, but english-russian works the same way. Just another tokens)

The input(source) sentence passes through the encoder, collects hidden states and then goes into decoder model. The first initial hidden state of decoder model is the last hidden state of the encoder model. `<sos>` and `<eos>` tokens mean that the sentence starts and sentence ends. 

Lets say we have <img src="https://render.githubusercontent.com/render/math?math=$X = \\{x_1, x_2, ... x_T\\}$"> , where <img src="https://render.githubusercontent.com/render/math?math=x_1 ="> `<sos>`, <img src="https://render.githubusercontent.com/render/math?math=x_2 ="> `guten` and etc. The initial hidden state, <img src="https://render.githubusercontent.com/render/math?math=h_0">, is usually either initialized to zeros or a learned parameter.

Once the final word, <img src="https://render.githubusercontent.com/render/math?math=x_T">, has been passed into the RNN via the embedding layer, we use the final hidden state, <img src="https://render.githubusercontent.com/render/math?math=h_T">, as the context vector, i.e. <img src="https://render.githubusercontent.com/render/math?math=h_T = z">. This is a vector representation of the entire source sentence.

Now we have our context vector, <img src="https://render.githubusercontent.com/render/math?math=z">, we can start decoding it to get the output/target sentence, "good morning". Again, we append start and end of sequence tokens to the target sentence. At each time-step, the input to the decoder RNN (blue) is the embedding, <img src="https://render.githubusercontent.com/render/math?math=d">, of current word, <img src="https://render.githubusercontent.com/render/math?math=d(y_t)">, as well as the hidden state from the previous time-step, <img src="https://render.githubusercontent.com/render/math?math=s_(t-1)">, where the initial decoder hidden state, <img src="https://render.githubusercontent.com/render/math?math=s_0">, is the context vector, <img src="https://render.githubusercontent.com/render/math?math=s_0 = z = h_T">, i.e. the initial decoder hidden state is the final encoder hidden state. Thus, similar to the encoder, we can represent the decoder as:

<img src="https://render.githubusercontent.com/render/math?math=">
