import pickle
from time import time
import sys
from CIoTS import CausalTSGenerator

data_path = 'notebooks/ICML/icml_data_v2/'

dim, in_edges, tau, autocorr, data_length = eval(sys.argv[1])

runs = range(10)

for run in runs:
    generator = CausalTSGenerator(dimensions=dim, max_p=tau, data_length=data_length,
                                  incoming_edges=in_edges, autocorrelation=autocorr)
    start = time()
    ts = generator.generate()
    elapsed = time() - start

    with open(data_path + f't={tau}_d={dim}_in={in_edges}_autocorr={autocorr}_{run}.pickle', 'wb') as f:
        pickle.dump(generator, f)

    print(dim, in_edges, tau, autocorr, data_length, run, ': ', elapsed)
