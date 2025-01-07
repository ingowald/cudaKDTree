#!/bin/bash
# script that measures fcp and knn perf for paper

for method in cct cct-xd spatial-cct spatial-stackBased stackBased stackBased-xd stackFree stackFree-xd ; do
    echo "results for method ${method}" | tee fcp-and-knn-results-${method}.txt
done

for f in 1000 10000 100000 1000000 10000000 100000000 1000000000; do
    for method in cct cct-xd spatial-cct spatial-stackBased stackBased stackBased-xd stackFree stackFree-xd ; do
	echo "##############" | tee -a fcp-and-knn-results-${method}.txt
	echo "### running for num data points = $f"  | tee -a  fcp-and-knn-results-${method}.txt
	echo "*** fcp, unlimited query"  | tee -a  fcp-and-knn-results.txt
	./cukd_float4-fcp-${method} -nr 100 $f | grep queries  | tee -a  fcp-and-knn-results-${method}.txt
	echo "*** knn, unlimited query"  | tee -a  fcp-and-knn-results.txt
	./cukd_float4-knn-${method} -nr 100 $f | grep queries  | tee -a  fcp-and-knn-results-${method}.txt 
	echo "*** knn, max-range 0.01 query"  | tee -a  fcp-and-knn-results.txt
	./cukd_float4-knn-${method} -nr 100 $f -r 0.01 | grep queries  | tee -a  fcp-and-knn-results-${method}.txt
    done
done

