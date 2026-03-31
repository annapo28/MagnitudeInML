import sys
import numpy as np
import pandas as pd

def calculate_means(solver, density = False):
   if density:
      df = pd.read_csv(f"fig_3_{solver}.csv", sep = "\t")
      df = df.set_index(["avgdeg", "graph"])
   else:
      df = pd.read_csv(f"fig_1_{solver}.csv", sep = "\t")
      df = df.set_index(["nodes", "graph"])
   df["mean"] = df.mean(axis = 1)
   changed_cells = -1
   while changed_cells != 0:
      changed_cells = 0
      for i in range(1, 11):
         oddcell = (df[f"run{i}"] / df["mean"]) > 2
         df.loc[oddcell, f"run{i}"] = np.NaN
         changed_cells += oddcell.sum()
      df = df.drop("mean", axis = 1)
      df["mean"] = df.mean(axis = 1)
      sys.stderr.write(f"{solver}\t{changed_cells}\n")
   sys.stderr.write(f"\n")
   if density:
      df.reset_index().to_csv(f"fig_3_{solver}_agg.csv", sep = "\t", index = False)
   else:
      df.reset_index().to_csv(f"fig_1_{solver}_agg.csv", sep = "\t", index = False)

for solver in ("approxchol", "augtree", "kmp", "cg", "base"):
   calculate_means(solver, density = False)
   calculate_means(solver, density = True)
