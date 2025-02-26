#!/bin/bash
# script that measures fcp and knn perf for paper

#for method in cct cct-xd spatial-cct spatial-stackBased stackBased stackBased-xd stackFree stackFree-xd ; do
#    echo "results for method ${method}" | tee fcp-and-knn-results-${method}.txt
#done


for f in fcp knn-clamped knn-unlimited; do 
    echo "" > results-$f.txt
done

for N in 1000 10000 100000 1000000 10000000 100000000 1000000000; do
    echo "### running for N = $N"
    for f in fcp knn-clamped knn-unlimited; do 
	echo "############## N = $N, uniform" | tee -a results-$f.txt
    done
    for method in cct cct-xd spatial-cct spatial-stackBased stackBased stackBased-xd stackFree stackFree-xd ; do
	# ==================================================================
	# for clamping use radius 10000 - that's 1% of [0,1M] domain
	# we generate samples in
	./cukd_float2-fcp-${method} -nr 10 $N > tmp.fcp.txt
	./cukd_float2-knn-${method} -nr 10 $N > tmp.knn-unlimited.txt
	./cukd_float2-knn-${method} -nr 10 $N -r 10000 > tmp.knn-clamped.txt
	
	for f in fcp knn-clamped knn-unlimited; do
	    stats=`cat tmp.$f.txt | grep NICE_STATS | awk '{print \$2}'`
	    perf=`cat tmp.$f.txt | grep "that is" | awk '{print \$3}'`
	    echo "stats $stats perf $perf"
	    echo "method $method stats $stats perf $perf" | tee -a results-$f.txt
	done
    done
    for f in fcp knn-clamped knn-unlimited; do 
	echo "############## N = $N, clustered" | tee -a results-$f.txt
    done
    for method in cct cct-xd spatial-cct spatial-stackBased stackBased stackBased-xd stackFree stackFree-xd ; do
	# ==================================================================
	# for clamping use radius 10000 - that's 1% of [0,1M] domain
	# we generate samples in
	./cukd_float2-fcp-${method} --clustered -nr 10 $N > tmp.fcp.txt
	./cukd_float2-knn-${method} --clustered -nr 10 $N > tmp.knn-unlimited.txt
	./cukd_float2-knn-${method} --clustered -nr 10 $N -r 10000 > tmp.knn-clamped.txt
	
	for f in fcp knn-clamped knn-unlimited; do
	    stats=`cat tmp.$f.txt | grep NICE_STATS | awk '{print \$2}'`
	    perf=`cat tmp.$f.txt | grep "that is" | awk '{print \$3}'`
	    echo "stats $stats perf $perf"
	    echo "method $method stats $stats perf $perf" | tee -a results-$f.txt
	done

    done
done

