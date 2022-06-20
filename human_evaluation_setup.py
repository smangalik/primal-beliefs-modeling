import pandas as pd
import random
import numpy as np

# read file 
generated_data_df = pd.read_csv("results_texts.csv", index_col = 0, header = 0)
tokens_generated_data_df = pd.read_csv("results_words_generated.csv", index_col = 0, header = 0)
tokens_lda_data_df = pd.read_csv("results_words_lda.csv", index_col = 0, header = 0)
print(generated_data_df.head(30))
print(tokens_generated_data_df.head(30))
print(tokens_lda_data_df.head(30))

# process data
tokens_generated_data_df = tokens_generated_data_df.iloc[:,:5]  # get first 5 frequent words only
tokens_lda_data_df = tokens_lda_data_df.iloc[:,:5]  # get first 5 frequent words only


# change swear words to ***
def remove_swears(df):
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            text = df.iloc[i,j]
            text = text.replace("fuck","f**k")
            text = text.replace("shit","s**t")
            text = text.replace("nigga","n***a")
            text = text.replace("dick","d*ck")
            df.iloc[i,j] = text
    return df
generated_data_df = remove_swears(generated_data_df) 
tokens_generated_data_df = remove_swears(tokens_generated_data_df) 
tokens_lda_data_df = remove_swears(tokens_lda_data_df) 



# intruding method
def intrude(data, n = 5):

    n_dimensions = len(data)
    n_generated = len(data.columns)

    # create empty intruded_data
    intruded_data = pd.DataFrame(np.zeros([n_dimensions, n]), index = data.index, columns = range(n))
    intruded_positions = np.zeros([n_dimensions,1])

    # fill in intruded_data
    for i in range(n_dimensions):
        intruded_data_onedimension = [data.iloc[i,:][j] for j in random.sample(range(n_generated), n)]
        instruded_position = random.sample(range(n), 1)[0]
        intruding_dimension = random.sample(list(range(0,i)) + list(range(i+1,n_dimensions)), 1)[0]
        intruding_position = random.sample(range(n_generated), 1)[0]
        # intrude
        intruded_data_onedimension[instruded_position] = data.iloc[intruding_dimension, :][intruding_position]
        intruded_data.iloc[i,:] = intruded_data_onedimension
        intruded_positions[i] = str(int(instruded_position))
        
    return intruded_data, intruded_positions


# run intruding method 
random.seed(0)
intruded_generated_data_df, intruded_positions_generated_data = intrude(generated_data_df, 5)
random.seed(1)
intruded_tokens_generated_data_df, intruded_positions_tokens_generated_data = intrude(tokens_generated_data_df)
random.seed(2)
intruded_tokens_lda_data_df, intruded_positions_tokens_lda_data = intrude(tokens_lda_data_df)
# readjust number to start from 1 rather 0 (ADDING 1 to every dimension)
intruded_generated_data_df.index = [('dimension_' + str(i+1)) for i in range(intruded_generated_data_df.shape[0])]
intruded_tokens_generated_data_df.index = [('dimension_' + str(i+1)) for i in range(intruded_tokens_generated_data_df.shape[0])]
intruded_tokens_lda_data_df.index = [('dimension_' + str(i+1)) for i in range(intruded_tokens_lda_data_df.shape[0])]
intruded_positions_generated_data = [(item + 1) for item in intruded_positions_generated_data]
intruded_positions_tokens_generated_data = [(item + 1) for item in intruded_positions_tokens_generated_data]
intruded_positions_tokens_lda_data = [(item + 1) for item in intruded_positions_tokens_lda_data]
# print out
for i in range(50):
    print(" ========= dimension {} ========= ".format(str(i+1)))
    print(" === texts generated === ")
    print(intruded_generated_data_df.iloc[i,:])
    print(" intruded position is: {} {} ".format("".join([" "]*100), str(intruded_positions_generated_data[i]))) 
    print(" === tokens generated frequencies === ")
    print(intruded_tokens_generated_data_df.iloc[i,:])
    print(" intruded position is: {} {} ".format("".join([" "]*100), str(intruded_positions_tokens_generated_data[i]))) 
    print(" === tokens lda frequencies === ")
    print(intruded_tokens_lda_data_df.iloc[i,:])
    print(" intruded position is: {} {} ".format("".join([" "]*100), str(intruded_positions_tokens_lda_data[i]))) 
    print("\n\n")


# save to file
intruded_generated_data_df.transpose().to_csv("/data/hvu/bert-exploration/finetuned_model/gpt_wrapper_test_50/results_texts_humaneval.csv")
pd.DataFrame(intruded_positions_generated_data, index = intruded_generated_data_df.index).to_csv("/data/hvu/bert-exploration/finetuned_model/gpt_wrapper_test_50/results_texts_humaneval_solution.csv")
intruded_tokens_generated_data_df.transpose().to_csv("/data/hvu/bert-exploration/finetuned_model/gpt_wrapper_test_50/results_words_generated_humaneval.csv")
pd.DataFrame(intruded_positions_tokens_generated_data, index = intruded_tokens_generated_data_df.index).to_csv("/data/hvu/bert-exploration/finetuned_model/gpt_wrapper_test_50/results_words_generated_humaneval_solution.csv")
intruded_tokens_lda_data_df.transpose().to_csv("/data/hvu/bert-exploration/finetuned_model/gpt_wrapper_test_50/results_words_lda_humaneval.csv")
pd.DataFrame(intruded_positions_tokens_lda_data, index = intruded_tokens_lda_data_df.index).to_csv("/data/hvu/bert-exploration/finetuned_model/gpt_wrapper_test_50/results_words_lda_humaneval_solution.csv")





















