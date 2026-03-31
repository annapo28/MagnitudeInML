using Graphs, SimpleWeightedGraphs
include("library/NetworkDistance.jl");

n = parse(Int, ARGS[1]);
method = ARGS[2];
graph = ARGS[3];

if graph == "er"
   G = adjacency_matrix(SimpleWeightedGraph(erdos_renyi(n, n * 2)));
elseif graph == "ba"
   G = adjacency_matrix(SimpleWeightedGraph(barabasi_albert(n, 2)));
elseif graph == "ws"
   G = adjacency_matrix(SimpleWeightedGraph(watts_strogatz(n, 4, 0.01)));
elseif graph == "sbm"
   k = trunc(Int, n / 4);
   G = adjacency_matrix(SimpleWeightedGraph(stochastic_block_model(reshape([k1 == k2 ? 3.12 : 0.3 for k2 in 1:4 for k1 in 1:4], (4, 4)), [k, k, k, k])));
end

s = rand(Float64, n);
t = rand(Float64, n);

dist = NetworkDistance.ge(G, s - t, method);
