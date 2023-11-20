
# nanoGPT + KV Cache


This repository is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). The code for the GPT model has been modfied to support a basic KV Cache in order to speed-up inference. Even a simple implementation of the KV Cache dramatically speeds up inference. The speed-up achieved increases with increase in Decoder Layers as we can cache in every layer. For a detailed explanation of the working of a KV Cache, read the KV Cache section below.  


## Jump To
* <a id="jumpto"></a> [Install](#install-)
* <a id="jumpto"></a> [KV Cache](#kv-cache-)
* <a id="jumpto"></a> [Inference](#inference-)
* <a id="jumpto"></a> [Results](#results-)
* <a id="jumpto"></a> [Possible Improvements](#possible-improvements-)
* <a id="jumpto"></a> [References](#references-)

**Note**- To isolate the speed-up achieved through a KV Cache, support for Flash Attention in the original repository has been disabled.
# Install [`↩`](#jumpto)
```
git clone https://github.com/rajlm10/nanoGPT_KV_Cache.git
cd nanoGPT_KV_Cache
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

# KV Cache [`↩`](#jumpto)
TODO


# Inference [`↩`](#jumpto)
To run the original implementation without a KV Cache
```
!python sample.py --init_from=gpt2 --max_new_tokens=1024 --start="What is the answer to life, the universe, and everything?" --num_samples=10 
```
To run the modified implementation with a KV Cache
```
!python modified_sample.py --init_from=gpt2 --max_new_tokens=1024 --start="What is the answer to life, the universe, and everything?" --num_samples=10 
```
Providing no start flag also works, the average generation time taken per sample is printed at the end along with all the generated samples.

**Choose model**- The init_from flag allows loading pretrained HuggingFace checkpoints and supports gpt2, gpt-medium, gpt2-large


# Results [`↩`](#jumpto)
TODO

# Possible Improvements [`↩`](#jumpto)
TODO

# References [`↩`](#jumpto)
TODO
