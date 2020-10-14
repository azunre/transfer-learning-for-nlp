# Transfer Learning for Natural Language Processing
Companion repository to Paul Azunre's "Transfer Learning for Natural Language Processing" Book

Rendered Jupyter notebooks are organized in folders by Chapter, with each folder containing a corresponding `requirements.txt` file representing Kaggle docker image pip dependency dump at the time of their latest succesful run by the author.

Ideally, run these directly on Kaggle, where notebooks are already hosted. This will require little to no setup on your part. Be sure to hit `Copy and Edit Kernel` at the top right of each Kaggle kernel page (after creating an account) to get going right away. 

NOTE: If you just copy and paste code into a new kernel, instead of taking the recommended `Copy and Edit Kernel` approach, you may face issues with dependencies as you will be starting from a different set of pre-installed Kaggle dependencies!

Alternatively, consider installing Anaconda locally and running the notebooks that way.

Should you decide to convert the notebooks to `.py` files for running them locally, you will likely deal with a lot installation debugging, etc. While this is a great learning experience everyone should try at least once in their career, do so at your own risk :-)

The following is an evolving list of notebooks that have been hosted so far, their corresponding Chapters and Kaggle links. This list will naturally grow longer as more chapters of this MEAP release are completed.


| Chapter  | Section(s) | Description | Kaggle link 
|-------------|-------------|-------------|-------------|
| 2 | 2.1-2.4 | Linear & Tree-based models for Email Sentiment Classification | https://www.kaggle.com/azunre/tl-for-nlp-section2-1-2-4-emails
| 2 | 2.1-2.4 | Linear & Tree-based models for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tl-for-nlp-section2-1-2-4-movies
| 2 | 2.5 | ELMo for Email Semantic Classification | https://www.kaggle.com/azunre/tl-for-nlp-section2-5-emails-elmo
| 2 | 2.5 | ELMo for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tl-for-nlp-section2-5-movies-elmo
| 2 | 2.6 | BERT for Email Semantic Classification | https://www.kaggle.com/azunre/tl-for-nlp-section2-6-emails-bert
| 2 | 2.6 | BERT for IMDB Movie Review Classification | https://www.kaggle.com/azunre/tl-for-nlp-section2-6-movies-bert
| 3 | 3.1 | IMDB Review Classification with word2vec and FastText | https://www.kaggle.com/azunre/tl-for-nlp-section3-1-movies-word-embeddings
| 3 | 3.2 | IMDB Review Classification with sent2vec | https://www.kaggle.com/azunre/tl-for-nlp-section3-2-movies-sentence-embeddings
| 3 | 3.3 | Dual Task Learning of IMDB and spam detection | https://www.kaggle.com/azunre/tl-for-nlp-section3-3-multi-task-learning
| 3 | 3.4 | Domain adaptation of IMDB classifier to new domain of Book Review Classification | https://www.kaggle.com/azunre/tl-for-nlp-section3-4-domain-adaptation
| 4 | 4.1 & 4.3 | Using SIMOn for column data type classification on baseball and BC library OpenML datasets | https://www.kaggle.com/azunre/tl-for-nlp-section4-1-4-3-column-type-classifier
| 4 | 4.2 & 4.4 | Using ELMo for "fake news" detection/classification | https://www.kaggle.com/azunre/tl-for-nlp-sections4-2-4-4-fake-news-elmo
| 5 | 5.1 | Transformer - self-attention visualization and English-Twi translation example with transformers library | https://www.kaggle.com/azunre/tl-for-nlp-section5-1
| 5 | 5.2 | GPT and DialoGPT in transformers library, application to chatbot and open-ended text generation | https://www.kaggle.com/azunre/tl-for-nlp-section5-2
| 5 | 5.3 | Applying BERT to filling-in-the-blanks and next sentence prediction | https://www.kaggle.com/azunre/tl-for-nlp-section5-3
| 5 | 5.4.2 | Fine-tuning mBERT on monolingual Twi data with pre-trained tokenizer | https://www.kaggle.com/azunre/tl-for-nlp-section5-4-2
| 5 | 5.4.3 | Fine-tuning mBERT on monolingual Twi data with tokenizer trained from scratch | https://www.kaggle.com/azunre/tl-for-nlp-section5-4-3
| 6 | 6.1 | Implementing ULMFiT adaptation strategies with fast.ai | https://www.kaggle.com/azunre/tl-for-nlp-section6-1
| 6 | 6.2 | Demonstrating advantages of Knowledge Distillation, i.e., DistilBERT, with Hugging Face Transformers library | https://www.kaggle.com/azunre/tl-for-nlp-section6-2
| 6 | 6.3 | Demonstrating advantages of cross-layer parameter sharing and embedding factorization, i.e., ALBERT, with Hugging Face Transformers | https://www.kaggle.com/azunre/tl-for-nlp-section6-3
| 6 | 6.4.2 | Fine-tuning BERT on a single GLUE task of measuring sentence similarity, i.e., STS-B | https://www.kaggle.com/azunre/tl-for-nlp-section6-4-2
| 6 | 6.4.3 | Fine-tuning BERT on a multiple GLUE tasks: sequential adaptation first to a data-rich question similarity scenario (QQP), followed by adaptation to a sentence similarity scenario (STS-B)| https://www.kaggle.com/azunre/tl-for-nlp-section6-4-3
| 6 | 6.5 | Using pretrained adapter modules with AdapterHub instead of fine-tuning| https://www.kaggle.com/azunre/tl-for-nlp-section6-5





To reiterate, just hit `Copy and Edit Kernel` at the top right of each Kaggle kernel page (after creating an account) to get going right away. Note that for GPU enabled notebooks, your **FREE** Kaggle GPU time is limited (to 30+ hours/week in February 2020, with the clock resetting at the end of each Friday). Be cautious and shut such notebooks down when not needed, when debugging non-GPU critical parts of the code, etc.

Kaggle  frequently  updates  the  dependencies,  i.e.,  versions  of  the  installed libraries on their docker images. To ensure that you are using the same dependencies as  we  did when  we  wrote  the  code – so  that  the  code  works  with  minimal  changes out-of-the-box – please make sure to select `Copy and Edit Kernel` for each notebook of  interest. If you  copy and paste the code into a new notebook and don’t follow this recommended process, you  may need  to adapt the code slightly for  the  specific  library versions installed for that notebook at the time you created it. 

This also applies if you elect to install a local environment.   For local installation, pay attention to the frozen dependency requirement list we have shared in the companion repo, which will guide you on which versions of libraries you will need. Finally, please note that while our aim is to update the code  to Tensorflow version >=2.0 syntax by the final release date  of  the  book, currently it is implemented in the more stable version <2.0 syntax.




