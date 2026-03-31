set terminal postscript eps enhanced color solid rounded linewidth 1
set datafile separator "\t"
set size 0.45,0.45
#set key bottom right
#set key samplen 1
unset key

set log y
set format y "10^{%L}"
set xlabel "Average Degree"
set ylabel "s"

set output "fig_3a.eps"
plot "fig_3_base_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_3_approxchol_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_3_augtree_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_3_kmp_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_3_cg_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_3b.eps"
plot "fig_3_base_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_3_approxchol_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_3_augtree_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_3_kmp_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_3_cg_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_3c.eps"
plot "fig_3_base_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_3_approxchol_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_3_augtree_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_3_kmp_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_3_cg_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_3d.eps"
plot "fig_3_base_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_3_approxchol_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_3_augtree_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_3_kmp_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_3_cg_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set log x
set format x "10^{%L}"
set xlabel "|V|"

set output "fig_2a.eps"
plot "fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "ER",\
"fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "BA",\
"fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "WS",\
"fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "SBM"

set output "fig_1a.eps"
plot "fig_1_base_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_1_approxchol_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_1_augtree_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_1_kmp_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "er"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_1b.eps"
plot "fig_1_base_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_1_approxchol_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_1_augtree_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_1_kmp_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "ba"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_1c.eps"
plot "fig_1_base_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_1_approxchol_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_1_augtree_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_1_kmp_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "ws"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_1d.eps"
plot "fig_1_base_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_1_approxchol_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#377eb8" t "ApproxChol",\
"fig_1_augtree_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#4daf4a" t "Aug Tree",\
"fig_1_kmp_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#984ea3" t "KMP",\
"fig_1_cg_agg.csv" u 1:(stringcolumn(2) eq "sbm"?$13:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set datafile separator " "
set xlabel "|V|"
set ylabel "kB"

set output "fig_2b.eps"
plot "fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "er"?$4-447536:NaN) w lines lw 4 lc rgb "#e41a1c" t "ER",\
"fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "ba"?$4-447536:NaN) w lines lw 4 lc rgb "#377eb8" t "BA",\
"fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "ws"?$4-447536:NaN) w lines lw 4 lc rgb "#4daf4a" t "WS",\
"fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "sbm"?$4-447536:NaN) w lines lw 4 lc rgb "#984ea3" t "SBM"

set output "fig_4a.eps"
plot "fig_2_4.csv" u 3:(stringcolumn(1) eq "base" && stringcolumn(2) eq "er"?$4-447536:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "er"?$4-447536:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_4b.eps"
plot "fig_2_4.csv" u 3:(stringcolumn(1) eq "base" && stringcolumn(2) eq "ba"?$4-447536:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "ba"?$4-447536:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_4c.eps"
plot "fig_2_4.csv" u 3:(stringcolumn(1) eq "base" && stringcolumn(2) eq "ws"?$4-447536:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "ws"?$4-447536:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"

set output "fig_4d.eps"
plot "fig_2_4.csv" u 3:(stringcolumn(1) eq "base" && stringcolumn(2) eq "sbm"?$4-447536:NaN) w lines lw 4 lc rgb "#e41a1c" t "Baseline",\
"fig_2_4.csv" u 3:(stringcolumn(1) eq "cg" && stringcolumn(2) eq "sbm"?$4-447536:NaN) w lines lw 4 lc rgb "#ff7f00" t "CG"
