import argparse

#question_num = 1,2,3 according to the arguemnt parser we want for each question
def create_hyperparameter_parser(question_num):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset_path', type=str, required=True, help="path to dataset")
    parser.add_argument('--train', dest='train', action='store_true', help="train the model and save it in a predetermined location.")
    parser.add_argument('--model-path', dest='model_path', type=str, default='./', help="path to save/load model, if not .")
    
    if question_num == 1 or question_num == 2:
        parser.add_argument('-n', dest='n_samples', required=True, help="number of timeseries to sample")
        
        #B
        if question_num == 2:
            parser.add_argument('-mae', dest='mae', action='store', required=True, help="enter mean absolute error")
    
    if question_num == 3:
        parser.add_argument('-q', dest='query_set', required=True, help="enter queryset path")
        parser.add_argument('-od', dest='output_path', required=True, help="enter output_dataset_file")
        parser.add_argument('-oq', dest='output_query_path', required=True, help="enter output_query_file")

    
    args = parser.parse_args()
    return args
