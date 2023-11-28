# import library
import time
import pandas as pd 
from sklearn.decomposition import NMF
import numpy as np
import argparse

# main function       
if __name__ == "__main__":

    # Start time
    start_time = time.time()

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--algorithm", default='NMF', type=str, required=True)
    parser.add_argument("--n", default=None, type=str, required=True)
    args = parser.parse_args()

    output_file = args.output_file
    embeddings_file = args.embeddings_file

    # read text file and BERT embeddings
    print("Reading embeddings files", embeddings_file)
    if embeddings_file.endswith(".csv"):
        embeddings = pd.read_csv(embeddings_file, header = 0, index_col = 0)
    elif embeddings_file.endswith(".json"):
        embeddings_df = pd.read_json(embeddings_file, lines=True)
        embeddings = embeddings_df['vector']
        embeddings = pd.DataFrame(embeddings.values.tolist(), index = embeddings.index) # expand lists to columns
    else:
        raise ValueError("--embeddings_file must be .csv or .json file.")
    print(embeddings_df.head())
    print("Embeddings shape:", embeddings.shape)


    ## run dimension reduction
    print("\nBegin dimension reduction...")
    algorithm = args.algorithm
    N = int(args.n)
    # running NMF algorithm 
    if algorithm.lower() == "nmf":

        # Subtract min value from embeddings
        processed_embeddings = embeddings.copy(deep=True)
        processed_embeddings = processed_embeddings - processed_embeddings.min()
        assert np.min(np.min(processed_embeddings)) >= 0

        # Run NMF
        nmf_model = NMF(n_components=N, init='random', random_state=0, max_iter = 500, verbose = False)
        results = nmf_model.fit_transform(processed_embeddings)
        print("Error from NMF Reconstruction: " + str(nmf_model.reconstruction_err_))

    
    # Add message_id and message column
    results_df = pd.DataFrame(results)
    results_df.insert(0, "message", embeddings_df['text'])
    results_df.insert(0, "message_id", embeddings_df.index)
    results_df.columns = ["message_id","message"] + list(range(results.shape[1]))

    # print results
    print("Reduced embedding df shape: " + str(results_df.shape))
    print(results_df.head())  

    # save NMF results to file
    print("Saving reduced dimensions to", output_file)
    results_df.to_csv(output_file, header = True, index = False)

    # End time in readable format
    elapsed_time = time.time() - start_time
    print("Elapsed time (H:M:S) was " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))