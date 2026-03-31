write(stderr, "Importing libraries...\n");
using Graphs, SimpleWeightedGraphs
include("library/NetworkDistance.jl");

n = 10000;

open("fig_3_base.csv", "a") do f
   write(f, "avgdeg\tgraph\trun1\trun2\trun3\trun4\trun5\trun6\trun7\trun8\trun9\trun10\n")
   for graph in ["er", "ba", "ws", "sbm"]
      for avgdeg in [1, 2, 4, 8, 16, 32, 64]
         write(f, "$(avgdeg)\t$(graph)\t");
         for r in 1:1
            write(stderr, "base\t$(avgdeg)\t$(r)\n");
            if graph == "er"
               G = adjacency_matrix(SimpleWeightedGraph(erdos_renyi(n, trunc(Int, round(n * avgdeg, digits = 0)))));
            elseif graph == "ba"
               G = adjacency_matrix(SimpleWeightedGraph(barabasi_albert(n, avgdeg)));
            elseif graph == "ws"
               G = adjacency_matrix(SimpleWeightedGraph(watts_strogatz(n, trunc(Int, avgdeg * 2), 0.01)));
            elseif graph == "sbm"
               k = trunc(Int, n / 4);
               G = adjacency_matrix(SimpleWeightedGraph(stochastic_block_model(reshape([k1 == k2 ? avgdeg * 1.5 : avgdeg / 6 for k2 in 1:4 for k1 in 1:4], (4, 4)), [k, k, k, k])));
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
   end
end

for method in ["approxchol", "augtree", "kmp", "cg"]
   open("fig_3_$(method).csv", "a") do f
      write(f, "avgdeg\tgraph\trun1\trun2\trun3\trun4\trun5\trun6\trun7\trun8\trun9\trun10\n")
      for graph in ["er", "ba", "ws", "sbm"]
         for avgdeg in [1, 2, 4, 8, 16, 32, 64]
            write(f, "$(avgdeg)\t$(graph)\t");
            for r in 0:10
               write(stderr, "$(method)\t$(avgdeg)\t$(r)\n");
               if graph == "er"
                  G = adjacency_matrix(SimpleWeightedGraph(erdos_renyi(n, trunc(Int, round(n * avgdeg, digits = 0)))));
               elseif graph == "ba"
                  G = adjacency_matrix(SimpleWeightedGraph(barabasi_albert(n, avgdeg)));
               elseif graph == "ws"
                  G = adjacency_matrix(SimpleWeightedGraph(watts_strogatz(n, trunc(Int, avgdeg * 2), 0.01)));
               elseif graph == "sbm"
                  k = trunc(Int, n / 4);
                  G = adjacency_matrix(SimpleWeightedGraph(stochastic_block_model(reshape([k1 == k2 ? avgdeg * 1.5 : avgdeg / 6 for k2 in 1:4 for k1 in 1:4], (4, 4)), [k, k, k, k])));
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
