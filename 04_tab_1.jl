write(stderr, "Importing libraries...\n");
using Laplacians
include("library/NetworkDistance.jl");

results = Dict{Int64, Dict{String, Float64}}();

for congress in [85, 105, 113]
   write(stderr, "$(congress)...\n");
   results[congress] = Dict{String, Float64}();
   for method in ["base", "approxchol", "augtree", "kmp", "cg"]
      G = readIJ("data/congress$(congress).edges");
      o = zeros(0);
      open("data/congress$(congress).nodes", "r") do f
         for ln in eachline(f)
            fields = split(ln, "\t");
            append!(o, parse(Float64, fields[3]));
         end
      end
      results[congress][method] = NetworkDistance.ge(G, o, method);
   end
end

open("tab_1.csv", "w") do f
   write(f, "method\t85th\t105th\t113th\n");
   write(f, "base\t$(results[85]["base"])\t$(results[105]["base"])\t$(results[113]["base"])\n");
   for method in ["approxchol", "augtree", "kmp", "cg"]
      write(f, "$(method)\t$(abs(results[85]["base"] - results[85][method]))\t$(abs(results[105]["base"] - results[105][method]))\t$(abs(results[113]["base"] - results[113][method]))\n");
   end
end

