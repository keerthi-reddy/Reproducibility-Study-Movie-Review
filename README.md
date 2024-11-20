# Reproducibility-Study-Movie-Review
Assignment-3

Karthik Reddy Musku - G01446785

Keerthi Ramireddy - G01450961

# Introduction:
This repository provides the implementation of the Select and Summarize model for abstractive summarization of long-form narrative texts, specifically movie scripts. Movie scripts are complex and lengthy, posing significant computational and memory challenges for traditional language models. To address these issues, this project introduces a two-stage summarization approach:

Scene Saliency Classification - Identifying the most critical "salient" scenes in the script using a transformer-based architecture.
Abstractive Summarization - Generating summaries based solely on the identified salient scenes, ensuring efficient and relevant content generation.
The approach leverages pre-trained models like RoBERTa, Longformer Encoder-Decoder (LED), and Pegasus-X to process and summarize lengthy scripts. It is validated using high-quality datasets like ScriptBase and MENSA, and evaluated using metrics such as ROUGE and BERTScore.

This repository contains code, datasets, and configurations for reproducing the results and extending the proposed methodology.

## Steps to run the code:
1. Install the packages using pip install -r requirements.txt
2. Run the precompute embeddings using python precompute_embeddings.py
3. Run the salient feature classification using python sal.py
4. Run the summarize model using python sum.py --fp16 --grad_ckpt --max_input_len=8192 
