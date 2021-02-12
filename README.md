# Transfer Learning for Natural Language Processing
Companion repository to [Paul Azunre's "Transfer Learning for Natural Language Processing" Book](https://www.manning.com/books/transfer-learning-for-natural-language-processing)

**Please note that this version of the repo follows a recent significant reordering of chapters. If you are looking for the original outdated ordering used during most of MEAP, please refer to [this repo version](https://github.com/azunre/transfer-learning-for-nlp/tree/57e3a316d51b7f7274ddff12be0fc3e0a2d77029)**

Rendered Jupyter notebooks are organized in folders by Chapter, with each folder containing a corresponding `kaggle_image_requirements.txt` file representing Kaggle docker image pip dependency dump at the time of their latest succesful run by the author. 

Please note that this requirements file is for the purpose of documenting and exactly replicating the environment on Kaggle on which the results reported in the book were achieved. If you try to configure an environment different from Kaggle -- say your local Apple machine -- by pip installing directly from this file, you will likely run into errors. In that case, this requirements file should only be used as a guide, and you can't expect it to work straight out of the box, due to many potential architecture-specific dependency conflicts. This isn't a usage mode we support, but it is definitely a good skill-building exercise to go through.

Ideally, run these notebooks directly on Kaggle, where notebooks are already hosted. This will require little to no setup on your part. Be sure to hit `Copy and Edit Kernel` at the top right of each Kaggle kernel page (after creating an account) to get going right away. 

NOTE: If you just copy and paste code into a new kernel, instead of taking the recommended `Copy and Edit Kernel` approach, you may face issues with dependencies as you will be starting from a different set of pre-installed Kaggle dependencies!

Alternatively, consider installing Anaconda locally and running the notebooks that way, potentially after converting to `.py` files if that is your preference.

In that case, heed the aforementioned caution about the provided `kaggle_image_requirements.txt` dependency files. You will likely deal with a lot installation debugging, etc. While this is a great learning experience everyone should try at least once in their career, do so at your own risk :-)

The following is a list of notebooks that have been hosted, their corresponding Chapters and Kaggle links.


| Chapter(s)  | Description | Kaggle link 
|-------------|-------------|-------------|
| 2-3 | Linear & Tree-based models for Email Sentiment Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-imdb-traditional
| 2-3 | Linear & Tree-based models for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-traditional
| 2-3 | ELMo for Email Semantic Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-elmo
| 2-3 | ELMo for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-imdb-elmo
| 2-3 | BERT for Email Semantic Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-spam-bert
| 2-3 | BERT for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapters2-3-imdb-bert
| 4 | IMDB Review Classification with word2vec and FastText | https://www.kaggle.com/azunre/tlfornlp-chapter4-imdb-word-embeddings
| 4 | IMDB Review Classification with sent2vec | https://www.kaggle.com/azunre/tlfornlp-chapter4-imdb-sentence-embeddings
| 4 | Dual Task Learning of IMDB and spam detection | https://www.kaggle.com/azunre/tlfornlp-chapter4-multi-task-learning
| 4 | Domain adaptation of IMDB classifier to new domain of Book Review Classification | https://www.kaggle.com/azunre/tlfornlp-chapter4-domain-adaptation
| 5-6 | Using SIMOn for column data type classification on baseball and BC library OpenML datasets | https://www.kaggle.com/azunre/tlfornlp-chapters5-6-column-type-classification
| 5-6 | Using ELMo for "fake news" detection/classification | https://www.kaggle.com/azunre/tlfornlp-chapters5-6-fake-news-elmo
| 7 | Transformer - self-attention visualization and English-Twi translation example with transformers library | https://www.kaggle.com/azunre/tlfornlp-chapter7-transformer
| 7 | GPT and DialoGPT in transformers library, application to chatbot and open-ended text generation | https://www.kaggle.com/azunre/tlfornlp-chapter7-gpt
| 8 | Applying BERT to filling-in-the-blanks and next sentence prediction | https://www.kaggle.com/azunre/tlfornlp-chapter8-bert
| 8 | Fine-tuning mBERT on monolingual Twi data with pre-trained tokenizer | https://www.kaggle.com/azunre/tlfornlp-chapter8-mbert-tokenizer-fine-tuned
| 8 | Fine-tuning mBERT on monolingual Twi data with tokenizer trained from scratch | https://www.kaggle.com/azunre/tlfornlp-chapter8-mbert-tokenizer-from-scratch
| 9 | Implementing ULMFiT adaptation strategies with fast.ai | https://www.kaggle.com/azunre/tlfornlp-chapter9-ulmfit-adaptation
| 9 | Demonstrating advantages of Knowledge Distillation, i.e., DistilBERT, with Hugging Face Transformers library | https://www.kaggle.com/azunre/tlfornlp-chapter9-distillation-adaptation
| 10 | Demonstrating advantages of cross-layer parameter sharing and embedding factorization, i.e., ALBERT, with Hugging Face Transformers | https://www.kaggle.com/azunre/tlfornlp-chapter10-albert-adaptation
| 10 | Fine-tuning BERT on a single GLUE task of measuring sentence similarity, i.e., STS-B | https://www.kaggle.com/azunre/tlfornlp-chapter10-multi-task-single-glue
| 10 | Fine-tuning BERT on a multiple GLUE tasks (*sequential adaptation*): first to a data-rich question similarity scenario (QQP), followed by adaptation to a sentence similarity scenario (STS-B)| https://www.kaggle.com/azunre/tlfornlp-chapter10-multi-task-glue-sequential
| 10 | Using pretrained adapter modules with AdapterHub instead of fine-tuning| https://www.kaggle.com/azunre/tlfornlp-chapter10-adapters
| Appendix B | Using Tensorflow V1, V2 and PyTorch to compute a function and its gradient via automatic differentiation | https://www.kaggle.com/azunre/tl-for-nlp-appendixb


To reiterate, just hit `Copy and Edit Kernel` at the top right of each Kaggle kernel page (after creating an account) to get going right away. Note that for GPU enabled notebooks, your **FREE** Kaggle GPU time is limited (to 30-40 hours/week in 2020, with the clock resetting at the end of each Friday). Be cautious and shut such notebooks down when not needed, when debugging non-GPU critical parts of the code, etc.

Kaggle  frequently  updates  the  dependencies,  i.e.,  versions  of  the  installed libraries on their docker images. To ensure that you are using the same dependencies as  we  did when  we  wrote  the  code – so  that  the  code  works  with  minimal  changes out-of-the-box – please make sure to select `Copy and Edit Kernel` for each notebook of  interest. If you  copy and paste the code into a new notebook and don’t follow this recommended process, you  may need  to adapt the code slightly for  the  specific  library versions installed for that notebook at the time you created it. 

This also applies if you elect to install a local environment.   For local installation, pay attention to the frozen dependency requirement list we have shared in the companion repo, which will guide you on which versions of libraries you will need. 

Finally, please note that while our initial aim in Chapters 2 and 3 was to write updated Tensorflow version >=2.0 syntax code, this is currently not possible for our experiment due to the dependency requirements of the `bert-tensorflow` package (see [this discussion](https://github.com/google-research/bert/issues/1140)). This is not that big of a deal, since we transition in later chapters from Tensorflow and Keras to the Hugging Face transformers library. That library uses latest dependencies (including Tensorflow >=2.0 in the backend, if you prefer it over PyTorch). In this context, you could view the exercise in Chapters 2-3, implemented in the more stable version <2.0 syntax, as a historical record of and experience with the initial packages that were developed for this problem.




