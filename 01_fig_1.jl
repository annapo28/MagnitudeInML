write(stderr, "Importing libraries...\n");
using Graphs, SimpleWeightedGraphs
include("library/NetworkDistance.jl");

open("fig_1_base.csv", "a") do f
   write(f, "nodes\tgraph\trun1\trun2\trun3\trun4\trun5\trun6\trun7\trun8\trun9\trun10\n")
   for graph in ["er", "ba", "ws", "sbm"]
      for n in [100, 200, 500, 1000, 2000, 5000]
         write(f, "$(n)\t$(graph)\t");
         for r in 0:10
            write(stderr, "base\t$(n)\t$(r)\n");
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
            trial = @timed NetworkDistance.ge(G, s - t, "base");
            if r != 0
               write(f, "$(trial.time)");
               if r < 10
                  write(f, "\t");
               end
            end
         end
         write(f, "\n");
         flush(f);
      end
      n = 10000
      write(stderr, "base\t$(n)\t0\n");
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
      trial = @timed NetworkDistance.ge(G, s - t, "base");
      write(f, "$(n)\t$(graph)\t$(trial.time)\n");
      flush(f);
   end
end

for method in ["approxchol", "augtree", "kmp", "cg"]
   open("fig_1_$(method).csv", "a") do f
      write(f, "nodes\tgraph\trun1\trun2\trun3\trun4\trun5\trun6\trun7\trun8\trun9\trun10\n")
      for graph in ["er", "ba", "ws", "sbm"]
         for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
            write(f, "$(n)\t$(graph)\t");
            for r in 0:10
               write(stderr, "$(method)\t$(n)\t$(r)\n");
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
               trial = @timed NetworkDistance.ge(G, s - t, method);
               if r != 0
                  write(f, "$(trial.time)");
                  if r < 10
                     write(f, "\t");
                  end
               end
            end
            write(f, "\n");
            flush(f);
         end
      end
   end
end
