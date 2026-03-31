write(stderr, "Importing libraries...\n");
using Laplacians, BenchmarkTools, SparseArrays
include("library/NetworkDistance.jl");

results = Dict{String, Vector{Float64}}();

for network in ["hiring_history", "eu_airlines", "eu_core", "openflights", "last_fm", "wiki_rfa", "fly_brain", "twitter15m", "patents"]
   write(stderr, "$(network)...\n");
   G = readIJ("data/$(network).edges");
   a = zeros(0);
   b = zeros(0);
   open("data/$(network).nodes", "r") do f
      for ln in eachline(f)
         fields = split(ln, ",");
         append!(a, parse(Float64, fields[2]));
         append!(b, parse(Float64, fields[3]));
      end
   end
   nodes = G.n;
   edges = sum(G) / 2;
   density = edges / ((nodes * (nodes - 1)) / 2);
   solver_runtime = @benchmark NetworkDistance.ge(G, a - b, "approxchol") samples = 1  evals = 1 setup = (a = $a; b = $b; G = $G);
   if nodes < 25000
      exact_runtime = @benchmark NetworkDistance.ge(G, a - b, "base") samples = 1  evals = 1 setup = (a = $a; b = $b; G = $G);
      results[network] = [nodes, edges, density, mean(solver_runtime.times) / 1e9, mean(exact_runtime.times) / 1e9];
   else
      results[network] = [nodes, edges, density, mean(solver_runtime.times) / 1e9, -1];
   end
end

write(stderr, "dbpedia...\n");
G = readIJ("data/dbpedia.edges");

nodes = G.n;
edges = sum(G) / 2;
density = edges / ((nodes * (nodes - 1)) / 2);

a = sprandn(nodes, 0.01);
b = sprandn(nodes, 0.01);

solver_runtime = @benchmark NetworkDistance.ge(G, a - b, "approxchol") samples = 1  evals = 1 setup = (a = $a; b = $b; G = $G);
results["dbpedia"] = [nodes, edges, density, mean(solver_runtime.times) / 1e9, -1];

open("tab_2_fig_5.csv", "w") do f
   write(f, "network\tnodes\tedges\tdensity\truntime_solver\truntime_exact\n");
   for network in ["hiring_history", "eu_airlines", "eu_core", "openflights", "last_fm", "wiki_rfa", "fly_brain", "twitter15m", "patents", "dbpedia"]
      write(f, "$(network)\t$(results[network][1])\t$(results[network][2])\t$(results[network][3])\t$(results[network][4])\t$(results[network][5])\n");
   end
end

