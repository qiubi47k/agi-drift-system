#!/bin/bash
cd /app/data/所有对话/主对话/drift_system
nohup python3 -u auto_loop.py > /tmp/auto_loop_v5.log 2>&1 &
echo $!
