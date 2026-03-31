set terminal postscript eps enhanced color solid rounded linewidth 1
set datafile separator "\t"
set size 0.45,0.45
set key top left
#set key samplen 1
#unset key

set log xy
set format xy "10^{%L}"
set ylabel "s"

set xrange [1000:2e8]
f(x) = (1.446 * (10 ** -7)) * x ** 1.14
set xlabel "|E|"
set output "fig_5a.eps"
plot "tab_2_fig_5.csv" u 3:5 w points pt 7 lc rgb "#377eb8" notitle,\
f(x) w lines lc rgb "#377eb8" t 'y = x^{1.14}'

set xrange [100:30000]
g(x) = (1.21 * (10 ** -9)) * x ** 2.91
set xlabel "|V|"
set output "fig_5b.eps"
plot "tab_2_fig_5.csv" u 2:6 w points pt 7 lc rgb "#e41a1c" notitle,\
g(x) w lines lc rgb "#e41a1c" t 'y = x^{2.91}'
