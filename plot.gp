#!/usr/bin/env gnuplot

reset session

set term png size 1800,960
set output 'loss.png'
set grid

set style line 1 lc rgb '#C00000' lt 1 lw 2 pt 7 pi -1 ps 0.5
set style line 2 lc rgb '#00C000' lt 1 lw 2 pt 7 pi -1 ps 0.5
set style line 3 lc rgb '#0000C0' lt 1 lw 2 pt 7 pi -1 ps 0.5

set multiplot title 'YOLO'

set size 0.67,0.5
set origin 0,0.5
stats "< awk '$1==\"step#:\" {print $2, $8}' training.log" every ::1 using 1:2;
max_y = floor(STATS_min_y * 10);
set yrange [0:max_y]
set title 'loss'
set xlabel 'step'

plot "< awk '$1==\"step#:\" {print $2, $8}' training.log" using 1:2 title 'train' w lines ls 1

set size 0.33,0.5
set origin 0.67,0.5
set title 'learning rate'
set xlabel 'step'
set autoscale y
plot "< awk '$1==\"step#:\" {print $2, $5}' training.log" using 1:2 title '' w lines ls 1

set size 0.25,0.5

set origin 0,0
set title 'mAP\@0.5'
set yrange [0:1]
set xlabel 'epoch'
plot "< awk '$1==\"EPOCH\" {print $2, $3}' training.log" using 1:2 title '' w linespoints ls 1

set origin 0.25,0
set title 'macro'
plot "< awk '$1==\"EPOCH\" {print $2, $4}' training.log" using 1:2 title 'precision' w linespoints ls 3, \
     "< awk '$1==\"EPOCH\" {print $2, $5}' training.log" using 1:2 title 'recall' w linespoints ls 2, \
     "< awk '$1==\"EPOCH\" {print $2, $6}' training.log" using 1:2 title 'f1-score' w linespoints ls 1

set origin 0.5,0
set title 'micro'
plot "< awk '$1==\"EPOCH\" {print $2, $7}' training.log" using 1:2 title 'precision' w linespoints ls 3, \
     "< awk '$1==\"EPOCH\" {print $2, $8}' training.log" using 1:2 title 'recall' w linespoints ls 2, \
     "< awk '$1==\"EPOCH\" {print $2, $9}' training.log" using 1:2 title 'f1-score' w linespoints ls 1

set origin 0.75,0
set title 'weighted'
plot "< awk '$1==\"EPOCH\" {print $2, $10}' training.log" using 1:2 title 'precision' w linespoints ls 3, \
     "< awk '$1==\"EPOCH\" {print $2, $11}' training.log" using 1:2 title 'recall' w linespoints ls 2, \
     "< awk '$1==\"EPOCH\" {print $2, $12}' training.log" using 1:2 title 'f1-score' w linespoints ls 1
unset multiplot

pause 60; refresh; reread
