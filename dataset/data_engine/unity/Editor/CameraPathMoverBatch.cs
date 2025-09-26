using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;
using System.IO;

public static class CameraPathMoverBatch
{
    public static void PerformTask()
    {   
        int dataNum = 10;
        string outputDirectory = Path.Combine(Application.dataPath, "CameraOutputs");

        

        // ---------------------------------------------------------
        //   PARSE COMMAND LINE ARGUMENTS RIGHT HERE
        // ---------------------------------------------------------
        string[] args = System.Environment.GetCommandLineArgs();
        foreach (string arg in args)
        {
            // e.g. -dataNum=15
            if (arg.StartsWith("-dataNum="))
            {
                if (int.TryParse(arg.Substring("-dataNum=".Length), out int parsedDataNum))
                {
                    dataNum = parsedDataNum;
                }
            }

            // e.g. -outputDir="C:/SomePath"
            if (arg.StartsWith("-outputDir="))
            {
                // Remove the "-outputDir=" prefix
                outputDirectory = arg.Substring("-outputDir=".Length).Trim('"');
            }

            if (arg.StartsWith("-scene="))
            {
                // Remove the "-scene=" prefix
                string scenePath = arg.Substring("-scene=".Length).Trim('"');
                EditorSceneManager.OpenScene(scenePath,OpenSceneMode.Single);
            }

            // … parse other parameters as needed
        }

        // 2) Set up a camera
        GameObject cameraObj = new GameObject("Main Camera");
        cameraObj.tag = "MainCamera";
        Camera cam = cameraObj.AddComponent<Camera>();

        // 3) Add the CameraPathMover component
        CameraPathMover mover = cameraObj.AddComponent<CameraPathMover>();
        mover.dataNum = dataNum;
        mover.outputDirectory = outputDirectory;
        
        // ---------------------------------------------------------
        //   FALLBACKS OR DEFAULTS IF ARGS WERE NOT PROVIDED
        // ---------------------------------------------------------
        // If user didn’t pass -dataNum=, we can still default:
        if (mover.dataNum == 0)
            mover.dataNum = 10; 

        if (string.IsNullOrEmpty(mover.outputDirectory))
            mover.outputDirectory = Path.Combine(Application.dataPath, "CameraOutputs");

        // ---------------------------------------------------------
        //   CONTINUE CONFIGURING ANY OTHER FIELDS
        // ---------------------------------------------------------
        mover.frameStep = 0.4f;
        mover.outputWidth = 500;
        mover.outputHeight = 500;
        mover.navigationDistance = 100f;
        mover.rotationTimes = 4;
        mover.minX = -50f;
        mover.maxX = 55f;
        mover.minY = 1.6f;
        mover.maxY = 1.8f;
        mover.minZ = -55f;
        mover.maxZ = 53f;
        mover.maxSamplingAttempts = 100;
        mover.fixedLastSegmentLength = 20f;
        Debug.Log("CameraPathMoverBatch: Configured CameraPathMover component  and  start  running  the  process.");
        // 4) Now run mover.Start() in batch mode
        mover.RunProcess();

        // 5) Exit Unity when done
        EditorApplication.Exit(0);
    }
}