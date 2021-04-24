import argparse
import numpy as np
from utils import *
from pysr import pysr, best, best_callable

def main():
    print("Please edit this file to config the solver for yourself")
    parser = argparse.ArgumentParser()
    parser.add_argument("derivatives_path", help="path to the derivative features", type=str)
    parser.add_argument("dynamics_path", help="path to the dynamics target", type=str)
    parser.add_argument("-nor", "--normalize", help="normalize the derivative features", action="store_true")
    parser.add_argument("-iter", "--niterations", nargs='?', const=10, type=int)
    parser.add_argument("-path", "--save_path", nargs='?', const=None, type=str)
    args = parser.parse_args()
    derivatives = np.load(args.derivatives_path)
    dynamics = np.load(args.dynamics_path)
    if args.normalize:
        derivatives = numpy_minmax_normalize(derivatives)
    if len(dynamics.shape) > 1:
        dynamics = np.squeeze(dynamics)
    equations = None
    try:
        equations = pysr(derivatives, dynamics, niterations=args.niterations, binary_operators=["+", "-", "*", "/",], 
                        unary_operators=[], batching=True, warmupMaxsizeBy=15.0, procs=4, populations=20, npop=2000) 
    except KeyboardInterrupt:
        print("Detect KeyboardInterrupt! Stop the algorithm.")
        print("The best equation found:", print(best(equations)))
        df = equations.drop(labels='lambda_format', axis=1)
        if args.save_path is not None:
            print("Saving the resulted dataframe to", args.save_path)
            df.to_pickle(args.save_path)

if __name__ == '__main__':
    main()
