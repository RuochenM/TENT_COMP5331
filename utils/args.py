import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--test_epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate.')

    parser.add_argument('--weight_decay', type=float, default=5e-4,  # 5e-4
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--hidden2', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
                        
    return parser.parse_args()