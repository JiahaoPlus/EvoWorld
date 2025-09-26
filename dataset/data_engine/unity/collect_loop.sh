#!/bin/bash

UNITY_PATH='/home/lubote/Unity/Hub/Editor/6000.0.32f1/Editor/Unity'
TOTAL_FRAMES=$1
BATCH_SIZE=150
LOOPS=$((TOTAL_FRAMES / BATCH_SIZE))
REMAINDER=$((TOTAL_FRAMES % BATCH_SIZE))

if [ $REMAINDER -ne 0 ]; then
  echo "Warning: TOTAL_FRAMES ($TOTAL_FRAMES) is not divisible by BATCH_SIZE ($BATCH_SIZE)."
fi

for (( i=1; i<=LOOPS; i++ ))
do
  echo "Running batch #$i of $LOOPS with dataNum=$BATCH_SIZE..."

  "$UNITY_PATH"  \
    -batchmode \
    -projectPath "/home/lubote/unity_ws/projects/poly-city" \
    -executeMethod CameraPathMoverBatch.PerformTask \
    -dataNum=$BATCH_SIZE \
    -scene="/home/lubote/unity_ws/projects/poly-city/Assets/myscene.unity" \
    -outputDir="/home/lubote/unity_ws/output/loop_batchmode" \
    -logFile "/home/lubote/unity_ws/log/low-texture_run_${i}.txt" \
    -quit

  echo "Finished batch #$i."

  # -----------------------------------------------------
  # Release system memory caches (requires root privileges).
  # -----------------------------------------------------
  echo "Dropping caches to release memory..."
  sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
  # (If you don't want to be prompted for password every time,
  #  you might set up NOPASSWD in /etc/sudoers for this command.)
done

echo "All $LOOPS batch runs complete!"

