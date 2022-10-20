#!/bin/bash
# script that measures fcp and knn perf for paper

echo "results:" > fcp-and-knn-results.txt
for f in 1000 10000 100000 1000000 10000000 100000000 1000000000; do
    echo "##############" >> fcp-and-knn-results.txt
    echo "### running for num data points = $f" >> fcp-and-knn-results.txt
    echo "*** fcp, unlimited query" >> fcp-and-knn-results.txt
    ./cukd_test_float4-fcp -nr 100 $f | grep queries >> fcp-and-knn-results.txt
    echo "*** knn, unlimited query" >> fcp-and-knn-results.txt
    ./cukd_test_float4-knn -nr 100 $f | grep queries >> fcp-and-knn-results.txt
    echo "*** knn, max-range 0.01 query" >> fcp-and-knn-results.txt
    ./cukd_test_float4-knn -nr 100 $f -r 0.01 | grep queries >> fcp-and-knn-results.txt
done

