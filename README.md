# Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection (a.k.a., MalConv2)

This is the PyTorch code implementing the approaches from our AAAI 2021 paper [Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection](https://arxiv.org/abs/2012.09390). Using it, you can train the original MalConv model faster and using less memory 
than before. You can also train our new MalConv with “Global Channel Gating” (GCG), what allows MalConv to learn feature interactions from across the entire inputs. 

## Code Organization

This is research quality code that has gone through some quick edits before going online, and comes with no warranty. The rough outline of the files in this repo. 

### binaryLoader.py 

`binaryLoader.py` contains the functions we use for loading in a dataset of binaries, and supports un-gziping them on the fly to reduce IO costs. It also includes a sampler that is used to create batches of similarly sized files to minimize excess 
padding used during training. This assumes the input dataset is already in sorted order by file size. 

### checkpoint.py

This contains code used to perform gradient checkpointing for reduced memory usage. This is optional and generally not necessary for our MalConv* models now, but was used during experimentation. 

### LowMemConv.py 

LowMemConv is the base class that implementations extend to obtain the fixed-memory pooling we introduced. This is provided by `seq2fix` function, which does the work of applying the convolution in chunks, tracking the winners, and then grouping the 
winning slices to run over with gradient calculations on. 

The user extends `LowMemConvBase`, implementing the `processRange` function, which applies whatever convolutional strategy they desire to a range of bytes. The `determinRF` function is used to determine the receptive field size by iteratively testing 
for the smallest input size that does not error, so that we know how to size our chunk sizes later. 