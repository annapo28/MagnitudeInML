echo "method graph nodes memory" > fig_2_4.csv;

for n in 100 200 500 1000 2000 5000 10000; do
   echo $n;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n base er 2>&1);
   echo "base er" $n $m >> fig_2_4.csv;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n base ba 2>&1);
   echo "base ba" $n $m >> fig_2_4.csv;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n base ws 2>&1);
   echo "base ws" $n $m >> fig_2_4.csv;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n base sbm 2>&1);
   echo "base sbm" $n $m >> fig_2_4.csv;
done;

for n in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
   echo $n;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n cg er 2>&1);
   echo "cg er" $n $m >> fig_2_4.csv;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n cg ba 2>&1);
   echo "cg ba" $n $m >> fig_2_4.csv;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n cg ws 2>&1);
   echo "cg ws" $n $m >> fig_2_4.csv;
   m=$(/usr/bin/time -f"%M" julia 02a_memusage.jl $n cg sbm 2>&1);
   echo "cg sbm" $n $m >> fig_2_4.csv;
done;
