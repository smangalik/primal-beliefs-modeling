# import library
import pandas as pd 
from sklearn.decomposition import NMF
import numpy as np
import argparse

# main function
def main():
    
    # arguments
    print("Reading arguments.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--algorithm", default=None, type=str, required=True)
    parser.add_argument("--n", default=None, type=str, required=True)
    args = parser.parse_args()


    # read text file and BERT embeddings
    print("Reading embeddings files.")
    embeddings_file = args.embeddings_file
    embeddings = pd.read_csv(embeddings_file, header = 0, index_col = 0)
    print("embeddings size: " + str(embeddings.shape))
    print(embeddings.head())


    ## run dimension reduction
    print("Dimensions reduction.")
    algorithm = args.algorithm
    N = int(args.n)
    # running NMF algorithm 
    if algorithm == "nmf":
        def nmf_process_embeddings(data):
            data = data - data.min()
            assert np.min(np.min(data)) >= 0
            return data
        processed_embeddings = nmf_process_embeddings(embeddings)
        model = NMF(n_components=N, init='random', random_state=0, max_iter = 500, verbose = False)
        results = model.fit_transform(processed_embeddings)
        print("reconstruction_err_: " + str(model.reconstruction_err_))


    # save NMF results to file
    print("Saving data.")
    output_file = args.output_file
    results_df = pd.DataFrame(results)
    results_df.insert(0, "message_id", embeddings.index)
    results_df.columns = ["message_id"] + list(range(results.shape[1]))
    results_df.to_csv(output_file, header = True, index = False)
    print("results_df size: " + str(results_df.shape))
    print("results_df: ")
    print(results_df.head())

        
if __name__ == "__main__":
    main()        
