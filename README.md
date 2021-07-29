# Transfer Learning for Natural Language Processing
Companion repository to [Paul Azunre's "Transfer Learning for Natural Language Processing" Book](https://tinyurl.com/47p8zzrv)

## Preliminary Instructions

**Please note that this version of the repo follows a recent significant reordering of chapters. If you are looking for the original outdated ordering used during most of MEAP, please refer to [this repo version](https://github.com/azunre/transfer-learning-for-nlp/tree/57e3a316d51b7f7274ddff12be0fc3e0a2d77029)**

Watch [![this intro video](https://img.youtube.com/vi/WIerqDFc5JM/maxresdefault.jpg)](https://www.youtube.com/watch?v=WIerqDFc5JM) first.

Rendered Jupyter notebooks are organized in folders by Chapter in this repo, with each folder containing a corresponding `kaggle_image_requirements.txt` file representing Kaggle docker image pip dependency dump at the time of their latest succesful run by the author. 

Please note that this requirements file is for the purpose of documenting the environment on Kaggle on which the results reported in the book were achieved. If you try to configure an environment different from Kaggle -- say your local Apple machine -- by pip installing directly from this file, you will likely run into errors. In that case, this requirements file should only be used as a guide, and you can't expect it to work straight out of the box, due to many potential architecture-specific dependency conflicts. 

**MOST OF THESE REQUIREMENTS WOULD NOT BE NECESSARY FOR LOCAL INSTALLATION, and this isn't a usage mode we support -- even if it is certainly a good "real-world" skill-building exercise to go through.** If you do decide to work locally, we recommend Anaconda. 

Ideally, run these notebooks directly on Kaggle, where notebooks are already hosted. This will require little to no setup on your part. **Be sure to hit `Copy and Edit Kernel` at the top right of each Kaggle kernel page (after creating an account) to get going right away.**


## WARNING
Please make sure to **"COPY AND EDIT NOTEBOOK"** on Kaggle, i.e., *fork* it, to use compatible library dependencies! **DO NOT CREATE A NEW NOTEBOOK AND COPY+PASTE THE CODE** - this will use the latest pre-installed Kaggle dependencies at the time you create the new notebook, and the code will need to be modified to make it work. Also make sure internet connectivity is enabled on your notebook. See Appendix A of book for a tutorial introduction to various features of Kaggle.

## Companion Notebooks

The following is a list of notebooks that have been hosted, their corresponding Chapters and Kaggle links.


| Chapter(s)  | Description | Kaggle link 
|-------------|-------------|-------------|
| 2-3 | Linear & Tree-based models for Email Sentiment Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-imdb-traditional
| 2-3 | Linear & Tree-based models for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-traditional
| 2-3 | ELMo for Email Semantic Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-elmo
| 2-3 | ELMo for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-imdb-elmo
| 2-3 | BERT for Email Semantic Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-bert
| 2-3 | Tensorflow 2 BERT for Email Semantic Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-bert-tf2
| 2-3 | BERT for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-imdb-bert
| 4 | IMDB Review Classification with word2vec and FastText | https://www.kaggle.com/azunre/tlfornlp-chapter4-imdb-word-embeddings
| 4 | IMDB Review Classification with sent2vec | https://www.kaggle.com/azunre/tlfornlp-chapter4-imdb-sentence-embeddings
| 4 | Dual Task Learning of IMDB and spam detection | https://www.kaggle.com/azunre/tlfornlp-chapter4-multi-task-learning
| 4 | Domain adaptation of IMDB classifier to new domain of Book Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapter4-domain-adaptation
| 5-6 | Using SIMOn for column data type classification on baseball and BC library OpenML datasets | https://www.kaggle.com/azunre/tlfornlp-chapters5-6-column-type-classification
| 5-6 | Using ELMo for "fake news" detection/classification | https://www.kaggle.com/azunre/tlfornlp-chapters5-6-fake-news-elmo
| 7 | Transformer - self-attention visualization and English-Twi translation example with transformers library | https://www.kaggle.com/azunre/tlfornlp-chapter7-transformer
| 7 | GPT and DialoGPT in transformers library, application to chatbot and open-ended text generation | https://www.kaggle.com/azunre/tlfornlp-chapter7-gpt
| 7, 11 | GPT-Neo (aims to be open GPT-3 "equivalent") in transformers library | https://www.kaggle.com/azunre/tlfornlp-chapter7-gpt-neo
| 8 | Applying BERT to filling-in-the-blanks and next sentence prediction | https://www.kaggle.com/azunre/tlfornlp-chapter8-bert
| 8 | Fine-tuning mBERT on monolingual Twi data with pre-trained tokenizer | https://www.kaggle.com/azunre/tlfornlp-chapter8-mbert-tokenizer-fine-tuned
| 8 | Fine-tuning mBERT on monolingual Twi data with tokenizer trained from scratch | https://www.kaggle.com/azunre/tlfornlp-chapter8-mbert-tokenizer-from-scratch
| 9 | Implementing ULMFiT adaptation strategies with fast.ai v1| https://www.kaggle.com/azunre/tlfornlp-chapter9-ulmfit-adaptation
| 9 | Implementing ULMFiT adaptation strategies with fast.ai v2| https://www.kaggle.com/azunre/tlfornlp-chapter9-ulmfit-adaptation-fast-aiv2
| 9 | Demonstrating advantages of Knowledge Distillation, i.e., DistilBERT, with Hugging Face Transformers library | https://www.kaggle.com/azunre/tlfornlp-chapter9-distillation-adaptation
| 10 | Demonstrating advantages of cross-layer parameter sharing and embedding factorization, i.e., ALBERT, with Hugging Face Transformers | https://www.kaggle.com/azunre/tlfornlp-chapter10-albert-adaptation
| 10 | Fine-tuning BERT on a single GLUE task of measuring sentence similarity, i.e., STS-B | https://www.kaggle.com/azunre/tlfornlp-chapter10-multi-task-single-glue
| 10 | Fine-tuning BERT on a multiple GLUE tasks (*sequential adaptation*): first to a data-rich question similarity scenario (QQP), followed by adaptation to a sentence similarity scenario (STS-B)| https://www.kaggle.com/azunre/tlfornlp-chapter10-multi-task-glue-sequential
| 10 | Using pretrained adapter modules with AdapterHub instead of fine-tuning| https://www.kaggle.com/azunre/tlfornlp-chapter10-adapters
| Appendix B | Using Tensorflow V1, V2 and PyTorch to compute a function and its gradient via automatic differentiation | https://www.kaggle.com/azunre/tl-for-nlp-appendixb


**To reiterate**, just hit `Copy and Edit Kernel` at the top right of each Kaggle kernel page (after creating an account) to get going right away. Note that for GPU enabled notebooks, your **FREE** Kaggle GPU time is limited (to 30-40 hours/week in 2020, with the clock resetting at the end of each Friday). Be cautious and shut such notebooks down when not needed, when debugging non-GPU critical parts of the code, etc.

Kaggle  frequently  updates  the  dependencies,  i.e.,  versions  of  the  installed libraries on their docker images. To ensure that you are using the same dependencies as  we  did when  we  wrote  the  code – so  that  the  code  works  with  minimal  changes out-of-the-box – please make sure to select `Copy and Edit Kernel` for each notebook of  interest. If you  copy and paste the code into a new notebook and don’t follow this recommended process, you  may need  to adapt the code slightly for  the  specific  library versions installed for that notebook at the time you created it. 

This also applies if you elect to install a local environment.   For local installation, pay attention to the frozen dependency requirement list we have shared in the companion repo, which will guide you on which versions of libraries you will need. Moreover, most of the requirements will not be necessary for local installation.

## Note About Tensorflow 1 versus Tensorflow 2

Finally, please note that because ELMo has not yet been ported to Tensorflow 2.x at the time of this  writing,  we  are  forced  to  use  Tensorflow  1.x  to  compare  it  fairly  to  BERT.  To confirm whether that is still the case you can check [the following link](https://tfhub.dev/s?q=elmo). We have however provided [a notebook illustrating of how to use BERT with Tensorflow 2.x for the spam classification example](https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-bert-tf2). We transition in later chapters from Tensorflow and Keras to the Hugging Face transformers library. That library uses latest dependendies in the backend, including Tensorflow 2.x if you prefer it over PyTorch. You could view the exercise in Chapters 2 and 3 as a historical record of and experience with  early  packages  that  were  developed  for  NLP  transfer  learning.  This  exercise  simultaneously  helps  you juxtapose Tensorflow 1.x with 2.x. Appendix B of the book will also help with this by outlining major differences between Tensorflow 1.x and 2.x.




