import argparse

#question_num = 1,2,3 according to the arguemnt parser we want for each question
def create_hyperparameter_parser(question_num):

    parser = argparse.ArgumentParser()
    if question_num == 1 or question_num == 2:
        parser.add_argument('-d', dest='dataset_path', type=str, help="path to dataset")
        parser.add_argument('-n', dest='n_samples', help="number of timeseries to sample")

        parser.add_argument('-l', dest='layers', action='store', help="enter number of layers")
        parser.add_argument('-s', dest='layer_size', action='store', help="enter size of layers")
        parser.add_argument('-e', dest='epochs', action='store', help="enter number of epochs")
        parser.add_argument('-b', dest='batch_size', action='store', help="enter batch size")
        #B
        if question_num == 2:
            parser.add_argument('-mae', dest='mae', action='store', help="enter dropout layers")
    
    if question_num == 3:
        parser.add_argument('-cl', dest='conv_layers', action='store', help="enter convolutional layers")
        parser.add_argument('-e', dest='epochs', action='store', help="enter number of epochs")
        parser.add_argument('-b', dest='batch_size', action='store', help="enter batch size")
        parser.add_argument('-l_dim', dest='latent_dim', action='store', help="enter latent dimension")
       
    
    args = parser.parse_args()
    return args
