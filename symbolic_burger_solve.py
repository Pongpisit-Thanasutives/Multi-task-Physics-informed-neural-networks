import argparse
import numpy as np
from utils import *
from preprocess import *
from models import *
from pysr import pysr, best, best_callable

def main():
    print("Please edit this file to config the solver for yourself")
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to the solver model", type=str)
    parser.add_argument("data_path", help="path to the burgers' equation dataset", nargs='?', const="./experimental_dataset/burgers_shock.mat", type=str)
    parser.add_argument("-sf", "--selected_features", nargs='?', const=None, type=str, default=None)
    parser.add_argument("-op", "--operations", nargs='?', const=None, type=str, default=None)
    parser.add_argument("-nor", "--normalize", help="normalize the derivative features", action="store_true")
    parser.add_argument("-iter", "--niterations", nargs='?', const=10, type=int, default=5)
    parser.add_argument("-path", "--save_path", nargs='?', const=None, type=str, default=None)
    args = parser.parse_args()
    semisup_model_state_dict = torch.load(args.model_path)
    
    network = Network(model=TorchMLP(dimensions=[2, 50, 50, 50 ,50, 50, 1], activation_function=nn.Tanh(), bn=nn.LayerNorm, dropout=None))
    selector = SeclectorNetwork(X_train_dim=6, bn=nn.LayerNorm)
    semisup_model = SemiSupModel(network=network, selector=selector, normalize_derivative_features=True, mini=None, maxi=None)
    semisup_model.load_state_dict(semisup_model_state_dict)

    print("Preparing dataset")
    X_star, u_star = get_trainable_data(*(space_time_grid(args.data_path, real_solution=True)))
    idx = np.random.choice(X_star.shape[0], 2000, replace=False)
    X_u_train = X_star[idx, :]; u_train = u_star[idx,:]
    derivatives, dynamics = semisup_model.network.get_selector_data(*dimension_slicing(to_tensor(X_star)))
    derivatives, dynamics = to_numpy(derivatives), to_numpy(dynamics).ravel()
    if args.selected_features is not None:
        derivatives = derivatives[:, list(map(int, args.selected_features.split()))]
    ops = ["+", "-", "*"]
    if args.operations is not None:
        ops = list(map(str, args.selected_features.split()))
    equations = None
    try:
        equations = pysr(derivatives, dynamics, niterations=args.niterations, binary_operators=ops, 
                        unary_operators=["inv(x)=1/x"], batching=True, procs=4, populations=20, npop=4000) 
                        
    except KeyboardInterrupt:
        print("Detect KeyboardInterrupt! Stop the algorithm.")
        print("The best equation found:", print(best(equations)))
        df = equations.drop(labels='lambda_format', axis=1)
        if args.save_path is not None:
            print("Saving the resulted dataframe to", args.save_path)
            df.to_pickle(args.save_path)

if __name__ == '__main__': main()
