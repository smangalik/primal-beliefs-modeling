# import libraries
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import numpy as np
import torch
import pickle 
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
import string

## architecture 1: embeddings fed in using transformation matrix for the second to last layer only 
# from gpt2_wrapper_arch1 import GPT2_WRAPPER 
# print("from gpt2_wrapper_arch1 import GPT2_WRAPPER")
## architecture 2: one same transformation matrix for all 12 layers
# from gpt2_wrapper_arch2 import GPT2_WRAPPER
# print("from gpt2_wrapper_arch2 import GPT2_WRAPPER")
## architecture 3: embeddings fed in using transformation matrix for the second to last layer only 
from gpt2_wrapper import GPT2_WRAPPER
print("from gpt2_wrapper import GPT2_WRAPPER")

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}

"""====================== METHODS DEFINITIONS ======================"""

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# truncating/padding method
def truncating_padding_sentence(tokens, block_size):
    if (len(tokens) > block_size):
        original_tokens_len = block_size
        tokens = tokens[:block_size]
    else:
        original_tokens_len = len(tokens)
        tokens = tokens + ["<|pad|> "]*(block_size - len(tokens))
    return tokens, original_tokens_len    

# creating attention mask method
def create_attention_mask(sentence_length, seq_length, gpt2_config, mask_type):
    # args:
    #   sentence_length is length of real text, from <|sos|>  to <|endoftext|>
    #   seq_length is length with <|pad|> (32, 64, 128, ...)
    
    if mask_type == "encoder_mask":
        print("Please set mask_type as: decoder_mask")
        return 
    if mask_type == "decoder_mask":
        # attention, the triangular matrix is [seq_length,seq_length+1] becasuse the original has one more "past" token
        mask_one_head = np.tril(np.ones([seq_length,seq_length+1]),1)
        mask_all_heads = [mask_one_head] * gpt2_config.n_head   
        mask_all_heads = np.array(mask_all_heads)
    return mask_all_heads            

def convert_tensor_inference(model, sentence_embedding, args, device, current_generated_sentences):
    

    # convert to tensor and put on device
    sentence_embedding_converted = torch.FloatTensor(sentence_embedding).unsqueeze(0).to(device)
    
    # generate sentence
    generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = sentence_embedding_converted, args = args, device = device)
    generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
    first_endoftext = generated_sample.find("<|endoftext|>") 
    generated_sample = str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)])
    count = 1
    while ((generated_sample in current_generated_sentences) and (count<10)):
        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = sentence_embedding_converted, args = args, device = device)
        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
        first_endoftext = generated_sample.find("<|endoftext|>") 
        generated_sample = str(generated_sample[:(first_endoftext)])
        count += 1
                
    current_generated_sentences.append(generated_sample)   

    # print generated sentence sample 
    generated_sample = generated_sample[6: ]
    print(str(generated_sample))
    
    return current_generated_sentences

def inference(model, args, device):
    print(" === inference ===")
    print("Task: generate text from each dimension")
    
    ### Get embeddings
    data_df = pd.read_csv(args.train_data_file, header = 0, index_col = 0)
    sentences_embeddings = data_df.iloc[:,1:].values
    sentences_text = data_df.iloc[:,:1]   

    print("Check embeddings data size: " + str(sentences_embeddings.shape))


    # analyzing, find std and mean of each hidden dimension    
    means = np.mean(sentences_embeddings, axis = 0)
    stds = np.std(sentences_embeddings, axis = 0)


    # percentages of values belonging to interval [mean + 2.std, mean + 4.std]
    percentages = []
    for i in range(len(means)):
        print("dimension {}:".format(str(i)))
        dimension_range = [means[i]+2.0*stds[i], means[i]+4*stds[i]]
        dimension_count = [ 1 if (item>=dimension_range[0] and item<=dimension_range[1]) else 0 for item in sentences_embeddings[:,i] ]
        dimension_percentage = np.sum(dimension_count)/len(sentences_embeddings[:,i])
        print(dimension_percentage)
        percentages.append(dimension_percentage)
    print("average percentages: " + str(np.mean(percentages)))


    # method for extracting most frequent words
    def most_frequent_words(sentences):
        # tokenize
        sentences = [sentence.split(" ") for sentence in sentences]
        # remove "the world is"
        sentences = [sentence[3:] for sentence in sentences]
        # merge all to one list
        all_words = []
        for sentence in sentences:
            all_words.extend(sentence)
        # remove stop words, remove marks (.,?|)
        processed_all_words = []
        for word in all_words:
            if word not in stopwords.words() and word not in string.punctuation:
                word = re.sub('['+string.punctuation+']', '', word)
                if word.strip()!="":
                    processed_all_words.append(word)
        all_words = processed_all_words
        # count frequencies
        df = pd.DataFrame(all_words, columns = ['word'])
        df_counts = df['word'].value_counts()
        return df_counts


    # set parameters
    explore_std_range = [2.0,4.0]
    std_step_interval = 0.2
    std_random_level = 0


    # generate sentence for each hidden dimension
    df_texts_generated = []
    df_texts_embeddings_csv = []
    hidden_size = means.shape[0]
    for i in range(hidden_size):
    
        print("=====")
        print("HIDDEN DIMENISON " + str(i) + ":")      

        ##### generating texts
        df_texts_generated_onedimension = []
        df_texts_embeddings_csv_onedimension = []
        print("***** generating text in interval: ")
        # explore the EXTREME positive direction
        for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
        
        
            print("samples around mean + {}*std:".format(round(std_position,1)))
            print("generation avoid repeating!")
            generated_samples = []   # avoid repeated generated_sample
            for _ in range(args.generate_num):     
                
                # sample embedding around embedding + std_position*stds[i]
                epsilon = np.random.uniform(-std_random_level,std_random_level)
                embedding_sample = np.copy(means)
                embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                torch_embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
        
                # generate sentence
                generated_count = 0    
                while True:
                    generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = torch_embedding_sample, args = args, device = device)
                    generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                    generated_count += 1
                    first_endoftext = generated_sample.find("<|endoftext|>") 
                    generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                    if (generated_sample_clean not in generated_samples) or generated_count >= 10:
                        generated_samples.append(generated_sample_clean)
                        break
                    
                # print generated sentence sample
                first_endoftext = generated_sample.find("<|endoftext|>") 
                cleaned_sentence = str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)])
                cleaned_sentence = cleaned_sentence.replace("<|sos|>","").replace("<|endoftext|>", "")
                print("generated_sample: <|sos|> " + cleaned_sentence + " <|endoftext|>")


                # add to list of one dimension
                df_texts_generated_onedimension.append(cleaned_sentence)
                df_texts_embeddings_csv_onedimension.append([cleaned_sentence, embedding_sample])

        # add to list of dimensions
        df_texts_generated.append(df_texts_generated_onedimension)
        df_texts_embeddings_csv.extend(df_texts_embeddings_csv_onedimension)

    ### save to file
    # texts generated 
    index = ['dimension_' + str(i) for i in range(hidden_size)]
    columns = ['generated_' + str(i+1) for i in range(len(df_texts_generated[0]))]
    df_texts_generated = pd.DataFrame(df_texts_generated, index = index, columns = columns)
    df_texts_generated.to_csv(args.output_dir + "/" + "results_texts.csv", index = True, header = True)
    # texts and embeddings csv
    df_texts_embeddings_csv = [[item[0]] + item[1].tolist() for item in df_texts_embeddings_csv]
    df_texts_embeddings_csv = pd.DataFrame(df_texts_embeddings_csv)
    print(df_texts_embeddings_csv)
    df_texts_embeddings_csv.insert(loc = 0, column = 'message_id', value = ['message_' + str(i) for i in range(len(df_texts_embeddings_csv))])
    df_texts_embeddings_csv.columns = ['message_id'] + ['message'] + ['dimension_' + str(i) for i in range(hidden_size)]
    df_texts_embeddings_csv.to_csv(args.output_dir + "/" + "results_texts_embeddings.csv", index = False, header = True)


    # count words frequencies 
    n = 10
    df_words_generated = pd.DataFrame(np.zeros([len(df_texts_generated), n]), index = df_texts_generated.index)
    for i in range(len(df_texts_generated)):
        words_counts = list(most_frequent_words(df_texts_generated.iloc[i,:]).index)
        words_counts = words_counts[:10]
        df_words_generated.iloc[i,:len(words_counts)] = words_counts

    # save to file
    print(df_words_generated.head())
    df_words_generated.to_csv(args.output_dir + "/" + "results_words.csv", index = True, header = True)


"""====================== MAIN FUNCTION ======================"""

# main function
def main():
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")                      
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    # model parameters
    parser.add_argument("--gpt2_model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--gpt2_model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--latent_size", default=-1, type=int, required=True,
                        help="Size of latent VAE layer.")    
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 
    
    # training parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
        
    # generating parameters
    parser.add_argument("--generate_num", type=int, default=None)
    parser.add_argument("--generate_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    # other generating parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # other parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")    
    
    # parsing parameters
    args = parser.parse_args()
    
    
    # =========== checking parameters and setting up  =========== #

    # setting things up    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")    # CHECK! make sure we use all 3 GPUs
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    # Set seed
    set_seed(args)


    # =========== bulilding model and inferencing  =========== #
    # Building model
    gpt2_config_class, gpt2_class, tokenizer_class = MODEL_CLASSES[args.gpt2_model_type]
    gpt2_config = gpt2_config_class.from_pretrained(args.gpt2_model_name_or_path, cache_dir = None)
    latent_size = args.latent_size
    model = GPT2_WRAPPER(gpt2_config, latent_size)
    
    # Load from checkpoint model
    model.from_pretrained(args)
    if args.block_size <= 0:  # modify args.block_size variable
        args.block_size = model.tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, model.tokenizer.max_len_single_sentence)
    
    # Send model to GPU
    model.to(args.device)    

    # Logging info
    logger.info("Inference parameters %s", args)
    

    # Inference
    args.gpt2_config = model.gpt2_config
    inference(model, args, device)    

    
if __name__ == "__main__":
    main()        




