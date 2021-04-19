from utils import VDCNN
from utils import make_dataloader, run_model
import torch
import argparse

def main(args):
    torch.manual_seed(0)

    train_fname = args.dataset_path + 'train.csv'
    test_fname = args.dataset_path + 'test.csv'

    dataloaders = make_dataloader(train_fname, test_fname)

    model = VDCNN(depth=args.depth, num_class=args.num_class)

    run_model(model, dataloaders, args.num_epochs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS547 Project 5: VDCNN')

    parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                         default = '../../data/yelp_review_polarity_csv/') 
    
    parser.add_argument('--num_epoch', dest='num_epochs', type=int,
                        default=15)

    parser.add_argument('--num_depth', dest='depth', type=int,
                        default=9)

    parser.add_argument('--num_class', dest='num_class', type=int,
                        default=2)

    args = parser.parse_args()
    main(args)
