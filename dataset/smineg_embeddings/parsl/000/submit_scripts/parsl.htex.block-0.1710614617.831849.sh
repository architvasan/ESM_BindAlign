
export JOBNAME=$parsl.htex.block-0.1710614617.831849
set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "1" == "1" ]] && echo "Found cores : $CORES"
WORKERCOUNT=1
FAILONANY=0
PIDS=""

CMD() {
process_worker_pool.py   -a 140.221.79.10 -p 0 -c 1.0 -m None --poll 10 --task_port=15144 --result_port=14456 --logdir=/lambda_stor/data/avasan/ESM_BindAlign/dataset/smineg_embeddings/parsl/000/htex --block_id=0 --hb_period=30  --hb_threshold=120 --cpu-affinity block --available-accelerators 0 1
}
for COUNT in $(seq 1 1 $WORKERCOUNT); do
    [[ "1" == "1" ]] && echo "Launching worker: $COUNT"
    CMD $COUNT &
    PIDS="$PIDS $!"
done

ALLFAILED=1
ANYFAILED=0
for PID in $PIDS ; do
    wait $PID
    if [ "$?" != "0" ]; then
        ANYFAILED=1
    else
        ALLFAILED=0
    fi
done

[[ "1" == "1" ]] && echo "All workers done"
if [ "$FAILONANY" == "1" ]; then
    exit $ANYFAILED
else
    exit $ALLFAILED
fi
