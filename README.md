# CIoTS
This repository implements an approach for *C*ausal *I*nference *o*n *T*ime *S*eries. It is heavily based on a paper by Chen, ["A time series causal model"](https://mpra.ub.uni-muenchen.de/24841/1/MPRA_paper_24841.pdf).  
It also contains the base implementation of Chen's algorithm and a simple time series data generator.

## Installation [![python version](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/)
```
pip install -U -r requirements.txt
```

## How to use

To start, first generate a causally dependent time series.  
Params:  
* `dimensions` (required) - the number of time series to generate 
* `max_p` (required) - the max time lag of any causal effects, i.e. for a point in time `t` there is at most a causal relationship `X<sub>t-p</sub> --> Y<sub>t</sub>`
* `data_length` - the number of data points to generate
* `incoming_edges` - the number of causal relationships per time series

```
from CIoTS import CausalTSGenerator

generator = CausalTSGenerator(dimensions=3, max_p=4, data_length=10000, incoming_edges=2)
data = generator.generate()
```

To visualize based on what model the dataset has been generated, we can visualize the partial time graph.

```
plt.title('Original graph')
generator.draw_graph()
```
![Original graph](https://user-images.githubusercontent.com/2228622/41195070-25e082cc-6c27-11e8-873f-16b003b7b998.png)

To estimate the graph, we run Chen's algorithm and render the result as graph.

```
from CIoTS import pc_chen, partial_corr_test, draw_graph

predicted_graph = pc_chen(partial_corr_test, data, p=4, alpha=0.05)

plt.title('Estimated graph')
draw_graph(predicted_graph, dimensions=3, max_p=4)
```
![Estimated graph](https://user-images.githubusercontent.com/2228622/41195069-25c32538-6c27-11e8-8abb-7d4205fb1a7d.png)

We can also print metrics of the algorithm's accuracy.
```
from CIoTS import evaluate_edges

pd.DataFrame(evaluate_edges(generator.graph, predicted_graph), index=[0])
```
![Metrics](https://user-images.githubusercontent.com/2228622/41195078-418f87c0-6c27-11e8-8fd5-9b1b719097a1.png)

The above tutorial runs Chen's base algorithm. Our research will focus on estimating `max_p` automatically and estimating the graph iteratively. This is still work in progress.
