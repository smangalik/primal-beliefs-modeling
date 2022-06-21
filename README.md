# primal-beliefs-modeling
Documents for the codes and data reproducing results in paper ["Modeling Latent Dimensions of Human Beliefs"](https://ojs.aaai.org/index.php/ICWSM/article/view/19358). 

## A. Codes and data structures

**(1) Source codes:**
 + `preprocess.py`: 
   steps for preprocessing tweets data. For example, reducing duplicates for long and highly repeatitive texts, processing symbols such as "\<newline>", "&amp", "&lt", etc, adding tweets ids, calculating data statistics, etc.
 + `extract_BERT_embeddings.py`: 
   codes for extracting BERT embeddings of tweets using the pre-trained provided BERT model. Each tweet in the data will be mapped to a BERT embeddings vector of size 1024.
 + `nmf_algorithm.py`: 
   codes running NMF factorization, factorizing tweets embeddings into compressed latent embeddings of size 50.
 + experimentally, we proposed 3 models architecture, implemented in these files:
	- `gpt2_wrapper_arch1.py`: 
          architecture 1 described in paper
	- `gpt2_wrapper_arch2.py`: 
          architecture 2 described in paper
	- `gpt2_wrapper.py`: 
          architecture 3 described in paper (also the main architecture)
 + `train_gpt2.py`: 
   codes for training the modified GPT-2 decoder model
 + `inference_gpt2.py`: 
   codes for generating texts of beliefs from latent dimensions
 

**(2) Data:**
 - `the_world_is_tweets.csv`: collected tweets data describing people's views about the world, which starting with "the world is...". The file has two columns 'id' and 'sentence' referring to the index and the tweet. Due to data privacy policy, this file is not made public.
 - `tweets_labels.csv`: file containing annotated tweets with primals classes by experts, used for prediction evaluation experiment.

**(3) Other relevant files:**
 - https://huggingface.co/gpt2/tree/main: link to original GPT-2 model weights which is read by train_gpt2.py to start training.



## B. Steps to reproduce proposed model
  
Below are the steps to build the model described in the paper.


**(1) Preprocessing texts:**
 + Details:
Preprocessing step in which we lowercase all tweets (to have them work most efficiently with large-uncased-BERT model), filtering out repeated quotes (to avoid unoriginal tweets) and cleaning data (replacing invalid symbols with correct ones, e.g. "&amp" or "&lt"). The following command specify the arguments to run the code. The file "the_world_is_tweets.csv" is provided in the Data directory.  
 + Command:
```
python3 preprocess.py \
	--input_file the_world_is_tweets.csv \
	--output_file tweets_processed.csv
```

**(2) Extracting BERT embeddings:**
 + Details:
In this step, we feed the processed data to BERT model to extract BERT embeddings. We used the large version of BERT, which has 24 transformers layers, 16 attention heads, 1024 hidden dimensions. Thorough details of the BERT model used can be found here in its original paper at [https://arxiv.org/abs/1810.04805]. The implementation and pretrained weights of the original BERT are downloaded from [https://huggingface.co/docs/transformers/model_doc/bert] (a widely used library for NLP transformers models). The embeddings of each sentence are computed by taking the means of the 4 layers of BERT, across all words in a sentence. 
 + Command:
```
python3 extract_BERT_embeddings.py \
	--input_file tweets_processed.csv \
	--output_file tweets_bert_embeddings.json \
	--model_type bert \
	--model_name_or_path bert-large-uncased \
	--do_lower_case \
 	--batch_size 8 \
	--block_size 32 \
	--text_column 1 \
	--id_column 0 \
	--layers "-1 -2 -3 -4" \
	--header True \
	--layers_aggregation mean 
```

**(3) Running NMF factorization:** 
 + Details:
The code below preprocesses data and run NMF algorithm to factorize embeddings data into 50 latent dimensions. We use scikit-learn implementation of NMF at [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html]. The hyper-parameters of NMF used is: tol=1e−4, max_iter=200, init="random", n_components = 5, other hyperparameters are kept as defaults.
 + Command:
```  
python3 nmf_algorithm.py \
	--embeddings_file tweets_bert_embeddings.csv \
	--output_file tweets_nmf50.csv \
	--algorithm nmf \
	--n 50
```

**(4) Training the modified GPT-2 model:**
 + Details:
Our method builds upon GPT-2 model, a widely-known text generative model with high performance. The GPT-2 version used in our paper is the base model, with 12 transformers layers, 12 attention heads, 768 hidden dimensions. As described in the main paper, we modified this model by adding transformation matrices that match the input vector size of 50 to the hidden vector size of 768. The total number of trainable parameters is 125 millions. Thorough details of GPT-2 can be found here in its original paper at [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]. The implementation and pretrained weights of the original GPT-2 are downloaded from [https://huggingface.co/docs/transformers/model_doc/gpt2] (a widely used library for NLP transformers models). The modification of GPT-2 is implemented in gpt2_wrapper.py file. The train_gpt2.py file is used to train the model using the following command.
 + Command:
```  
python3 train_gpt2.py \
	--train_data_file tweets_nmf50.csv \
	--output_dir trained_models/label_model \
	--gpt2_model_type gpt2 \
	--gpt2_model_name_or_path gpt2 \
	--latent_size 50 \
	--block_size 32 \
	--per_gpu_train_batch_size 16 \
	--gradient_accumulation_steps 1 \
	--do_train \
	--save_steps 500 \
	--num_train_epochs 5 \
	--overwrite_output_dir \
	--overwrite_cache
```

**(5) Inferencing (generating texts) with the trained model:**
 + Details:
The inference_gpt2.py file is used to generate texts from latent dimesions using the trained model from the step above. The argument generate_num indicates how many sentences to generate for each dimension. The arguments temperature, top_p and top_k are for nucleus sampling when generating texts, which control how broad or narrow the generated texts would be satistically. 
 + Command:
```  
python3 inference_gpt2.py \
	--train_data_file tweets_nmf50.csv \
	--output_dir trained_models/label_model \
	--gpt2_model_type gpt2 \
	--gpt2_model_name_or_path gpt2 \
	--latent_size 50 \
	--generate_num 5 \
	--generate_length 32 \
	--temperature 0.2 \
	--top_p 0.9 \
	--top_k 10 \
	--inference_test 1 \
	--method method_1 \
	--overwrite_cache
```


  
  
  
  
  
  
  
  
  
