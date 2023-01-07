# Transformer Architecture

## Overview

The Transformer is a neural network architecture that processes sequential input data using attention mechanisms. It was introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017 and has since become a widely used and influential model architecture in natural language processing tasks.

## Encoder

The input data is passed through the encoder, which processes the input and produces a continuous representation, or encoding, of the input data. Each encoder layer consists of two sub-layers: a multi-headed self-attention layer and a fully connected layer called a feedforward network (FFN). See the detailed explanation below for more information.

## Decoder

The decoder processes the encoding produced by the encoder and generates the output. The decoder has an additional attention layer that takes the encoder output as additional input. The decoder generates the output through a process of self-attention, attention over the encoder output, and FFN layers. See the detailed explanation below for more information.

## Attention Mechanisms

The attention mechanism allows the model to selectively focus on certain parts of the input and is key to the Transformer's ability to process long-range dependencies in the input data. In the multi-headed self-attention layers of the encoder and decoder, the attention layer has multiple "heads", each of which produces a weighted sum of the input. See the detailed explanation below for more information.

## Detailed Explanation

The Transformer consists of an encoder and a decoder, both of which are composed of multiple layers of attention and fully connected (fc) layers.

Encoder:
- The input is a sequence of tokens (e.g. words) represented as integer indices. 
- The input is passed through an embedding layer, which maps the integer indices to continuous dense vectors (the embedding). The embedding is learned during training and is used to capture the meaning of the input tokens in a continuous space.
- The input is then passed through a stack of encoder layers. Each encoder layer consists of two sub-layers:
  - A multi-headed self-attention layer, which computes the attention between all input tokens and generates a weighted sum of the input tokens (the "query" in attention mechanisms is the input tokens themselves). 
  - The attention layer has multiple "heads", each of which focuses on a different aspect of the input and produces a weighted sum. The outputs of the different heads are concatenated and passed through a final fc layer to produce the final attention output.
  - A fully connected layer that applies a non-linear transformation to the sum of the attention output and the input. This layer is called a feedforward network (FFN) and is used to further process the input data.
- The output of the encoder is a continuous representation of the input data, which is passed to the decoder.
![alt text](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_1.png)

Decoder:
- The decoder is similar to the encoder, but in addition to the attention and FFN layers, it also has an additional attention layer that takes the encoder output as additional input. This allows the decoder to "attend" to the encoder output at each step and incorporate context from the input sequence into the output sequence.
- The decoder generates the output through a process of self-attention (like in the encoder), attention over the encoder output, and FFN layers. At each step, the decoder produces an output based on the previous outputs (self-attention), the encoder output (encoder-decoder attention), and the input to the decoder (similar to the encoder-decoder attention but with the decoder input as the "query").
- The output of the decoder is a sequence of tokens that is intended to be a translation or summary of the input data.
![alt text](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_2.png)


## Attention Mechanisms

The attention mechanism allows the model to selectively focus on certain parts of the input and is key to the Transformer's ability to process long-range dependencies in the input data. In the multi-headed self-attention layers of the encoder and decoder, the attention layer has multiple "heads", each of which produces a weighted sum of the input. See the detailed explanation below for more information.
the multi-headed attention mechanism is used in the self-attention layers of the encoder and decoder. The attention mechanism allows the model to selectively focus on certain parts of the input and is key to the Transformer's ability to process long-range dependencies in the input data.

In the multi-headed attention mechanism, the attention layer has multiple "heads", each of which produces a weighted sum of the input. The input to the attention layer is a set of queries, keys, and values, and the output is a weighted sum of the values, where the weight for each value is computed using the dot product of the query with a key.

In the multi-headed attention mechanism, the queries, keys, and values are projected using separate linear transformations, and the attention is computed independently for each head. The outputs of the different heads are then concatenated and passed through a final linear transformation to produce the final attention output.

The multi-headed attention mechanism allows the model to attend to different aspects of the input in parallel, which can improve the model's ability to capture complex dependencies in the input data.
<img src="[image.png](https://machinelearningmastery.com/wp-content/uploads/2021/09/tour_3.png)" width="500">


