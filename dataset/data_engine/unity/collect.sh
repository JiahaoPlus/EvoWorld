
UNITY_PATH='/home/lubote/Unity/Hub/Editor/6000.0.32f1/Editor/Unity'
"$UNITY_PATH"  \
  -batchmode \
  -projectPath "/home/lubote/unity_ws/projects/poly-city" \
  -executeMethod CameraPathMoverBatch.PerformTask \
  -dataNum=200 \
  -scene="/home/lubote/unity_ws/projects/poly-city/Assets/myscene.unity" \
  -outputDir='/home/lubote/unity_ws/output/loop_batchmode' \
  -logFile "/home/lubote/unity_ws/log/low-texture.txt" \
  -quit
