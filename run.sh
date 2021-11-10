#!/bin/bash

date

export SENDCOUNT=1000000000

jsrun -n1 -a6 -g1 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./communications

date
