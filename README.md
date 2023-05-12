# Zero-shot Text Classification
1. Benchmarking zero-shot text classification models
2. Bi-encoder for zero-shot classification, a balance between speed & accuracy.



## Setup environment

OS: UNIX; Python version `3.8.10`; CUDA version `11.6`. 



Create conda environment: 

```bash
conda create -n zs-cls python=3.8.10 pip
```

At project root directory, install python packages: 

```bash
pip3 install -r requirements.txt
```



### Train Baseline

e.g. On GPT2 zero shot classification: 
```bash
export PYTHONPATH=$PATHONPATH:`pwd`
python3 zeroshot_classifier/baseline/gpt2.py
```

## Obsolete: Unified-Encoder

Exploring a unified framework for potentially many NLP tasks as encoding operations



Formalize common NLP tasks beyond Information Retrieval as 1) encoding then 2) simple operation, so that intermediate results can be cached. 

