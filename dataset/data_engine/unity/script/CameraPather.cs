using UnityEngine;
using System.IO;
using System;
using System.Linq;

public class CameraPathMover : MonoBehaviour
{
    public int dataNum = 10;             // Number of data collections
    public float frameStep = 0.4f; 
    public int outputWidth = 500;        // Output image width
    public int outputHeight = 500;       // Output image height
    public string outputDirectory = "";  // Output directory

    public float navigationDistance = 20f; // Total distance (n) for the camera path
    public int rotationTimes = 4;           // Number of segments (k) in the path

    public float minX = -50f;
    public float maxX = 55f;
    public float minY = 1.6f;
    public float maxY = 1.8f;
    public float minZ = -55f;
    public float maxZ = 53f;          // Boundaries for random position sampling
    public int maxSamplingAttempts = 100;   // Maximum attempts to find a valid path
    public float fixedLastSegmentLength = 12f; // Define fixed last segment length

    private int currentFrame = 0;
    private RenderTexture rt;
    private Texture2D screenShot;

    void Start()
    {
        RunProcess();  // calls the new method below
    }

    public void RunProcess()
    {
        // set rotation times to a random value between [2,3,4,5,10]
        
        rt = new RenderTexture(outputWidth, outputHeight, 24);
        screenShot = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);

        // 1) Identify the "base" folder that holds iteration subfolders
        //    i.e. something like:   <outputDirectory>/<navigationDistance>_<rotationTimes>
        string baseFolder = Path.Combine(
            outputDirectory,
            $"mixed_loop"
        );

        // Ensure the base folder exists
        if (!Directory.Exists(baseFolder))
        {
            Directory.CreateDirectory(baseFolder);
        }

        // 2) Count how many "Iteration_*" subfolders already exist
        //    so we know where to continue from.
        //    For safety, we only match subfolders that literally start with "Iteration_"
        //    If you have a different folder naming scheme, adjust the search pattern.
        string[] existingIterations = Directory.GetDirectories(baseFolder, "episode_*", SearchOption.TopDirectoryOnly);

        // currentCount = how many "Iteration_X" subfolders are already there
        int currentCount = existingIterations.Length;

        // 3) We'll generate an additional `dataNum` episodes,
        //    starting from `currentCount` up to `currentCount + dataNum - 1`
        int startIndex = currentCount;
        int endIndex   = currentCount + dataNum;

        // 4) Loop and generate episodes
        for (int i = startIndex; i < endIndex; i++)
        {
            rotationTimes = new int[] { 2, 3, 4, 5, 10 }[UnityEngine.Random.Range(0, 5)];
            if (rotationTimes == 2)
                {navigationDistance  = new float[] { 4f, 20f }[UnityEngine.Random.Range(0, 2)];}
            else
                {navigationDistance  = new float[] { 2f, 4f, 6f,10f,20f }[UnityEngine.Random.Range(0, 5)];}

            // The iteration folder for this new episode
            string iterationFolder = Path.Combine(baseFolder, $"episode_{i:D4}");

            // If for some reason it already exists, skip
            if (Directory.Exists(iterationFolder))
                continue;

            Directory.CreateDirectory(iterationFolder);

            // Sample a random start position
            Vector3 startPosition = SampleRandomStartPosition();

            // Try to generate a valid path
            PathSegment[] path = null;
            int attempts = 0;
            bool validPath = false;

            while (!validPath && attempts < maxSamplingAttempts)
            {
                attempts++;
                int numSteps = Mathf.RoundToInt(navigationDistance / frameStep);
                path = GenerateVectorsPrecise(numSteps, rotationTimes, frameStep, fixedLastSegmentLength);
                validPath = true;

                // Check validity
                Vector3 currentPosition = startPosition;
                foreach (var seg in path)
                {
                    Vector3 movementDirection = Quaternion.Euler(0, seg.Rotation, 0) * Vector3.forward;
                    Vector3 targetPosition = currentPosition + movementDirection * seg.Distance;

                    if (!IsPathSegmentValid(currentPosition, targetPosition))
                    {
                        validPath = false;
                        break;
                    }
                    currentPosition = targetPosition;
                }
            }

            // If no valid path after all attempts, skip
            if (!validPath) 
                continue;

            // Perform camera movement, capture images
            CaptureAndSaveImages(iterationFolder, startPosition, path);
        }
    }

    Vector3 SampleRandomStartPosition()
    {
        return new Vector3(
            UnityEngine.Random.Range(minX, maxX),
            UnityEngine.Random.Range(minY, maxY),
            UnityEngine.Random.Range(minZ, maxZ)
        );
    }

    PathSegment[] GenerateVectorsPrecise(
        int n,
        int k,
        float frameStep,
        float fixedLastSegmentLength,
        int? maxDistance = null,
        double learningRate = 0.005,
        int iterations = 10000,
        double maxError = 1e-5)
    {
        System.Random random = new System.Random();

        while (true)
        {
            // Random integer magnitudes for k segments
            int[] magnitudes = Enumerable.Repeat(1, k).ToArray();
            if (maxDistance == null) maxDistance = n / 2;
            int remainingDistance = n - k;

            // Increment random segments until total distance == n
            while (remainingDistance > 0)
            {
                int index = random.Next(0, k);
                if (magnitudes[index] < maxDistance)
                {
                    magnitudes[index]++;
                    remainingDistance--;
                }
            }

            // Normalize magnitudes
            double[] normalizedMagnitudes = magnitudes.Select(m => (double)m / n).ToArray();
            double[] angles = Enumerable.Range(0, k).Select(_ => random.NextDouble() * 2 * Math.PI).ToArray();

            // Try to make net x & y displacement ~ 0 via gradient descent
            for (int iterationCount = 0; iterationCount < iterations; iterationCount++)
            {
                double xSum = normalizedMagnitudes.Select((mag, i) => mag * Math.Cos(angles[i])).Sum();
                double ySum = normalizedMagnitudes.Select((mag, i) => mag * Math.Sin(angles[i])).Sum();

                // If sums are small enough, break
                if (Math.Abs(xSum) <= maxError && Math.Abs(ySum) <= maxError)
                    break;

                double[] gradX = normalizedMagnitudes.Select((mag, i) => -mag * Math.Sin(angles[i])).ToArray();
                double[] gradY = normalizedMagnitudes.Select((mag, i) =>  mag * Math.Cos(angles[i])).ToArray();

                for (int i = 0; i < k; i++)
                {
                    angles[i] -= learningRate * n * (xSum * gradX[i] + ySum * gradY[i]);
                }
            }

            // Check final sums
            double finalXSum = normalizedMagnitudes.Select((mag, i) => mag * Math.Cos(angles[i])).Sum();
            double finalYSum = normalizedMagnitudes.Select((mag, i) => mag * Math.Sin(angles[i])).Sum();

            // If small enough, build the path
            if (Math.Abs(finalXSum) <= maxError && Math.Abs(finalYSum) <= maxError)
            {
                PathSegment[] pathSegments = new PathSegment[k];

                // The original length of the last segment
                float originalLastSegmentLength = magnitudes[k - 1] * frameStep;

                for (int i = 0; i < k; i++)
                {
                    float previousAngleDeg = (i == 0) 
                        ? 0 
                        : (float)(angles[i - 1] * 180 / Math.PI);

                    float currentAngleDeg = (float)(angles[i] * 180 / Math.PI);

                    if (i == k - 1)
                    {
                        // Extend the last segment to fixedLastSegmentLength
                        pathSegments[i] = new PathSegment
                        {
                            Distance = fixedLastSegmentLength,
                            Rotation = Mathf.DeltaAngle(previousAngleDeg, currentAngleDeg)
                        };
                    }
                    else
                    {
                        pathSegments[i] = new PathSegment
                        {
                            Distance = magnitudes[i] * frameStep,
                            Rotation = Mathf.DeltaAngle(previousAngleDeg, currentAngleDeg)
                        };
                    }
                }

                return pathSegments;
            }
        }
    }

    bool IsPathSegmentValid(Vector3 startPosition, Vector3 endPosition)
    {
        Vector3 direction = (endPosition - startPosition).normalized;
        float distance = Vector3.Distance(startPosition, endPosition);
        // If raycast hits something, path is invalid
        return !Physics.Raycast(startPosition, direction, distance);
    }

    void CaptureAndSaveImages(string iterationFolder, Vector3 startPosition, PathSegment[] path)
    {
        transform.position = startPosition;
        transform.rotation = Quaternion.identity;

        string poseFilePath = Path.Combine(iterationFolder, "camera_poses.txt");
        using (StreamWriter poseWriter = new StreamWriter(poseFilePath))
        {
            poseWriter.WriteLine("Frame,Position.x,Position.y,Position.z,Rotation.x,Rotation.y,Rotation.z");

            float cumulativeAngle = 0;
            float frameInPath = 0;

            foreach (var segment in path)
            {
                cumulativeAngle += segment.Rotation;
                Quaternion segmentRotation = Quaternion.Euler(0, cumulativeAngle, 0);
                transform.rotation = segmentRotation;

                Vector3 movementDirection = transform.forward;
                int segmentFrames = Mathf.CeilToInt(segment.Distance / frameStep);

                for (currentFrame = 0; currentFrame < segmentFrames; currentFrame++)
                {
                    frameInPath += 1;
                    transform.position += movementDirection * frameStep;

                    Vector3 position = transform.position;
                    Vector3 eulerRotation = segmentRotation.eulerAngles;

                    // Record pose
                    poseWriter.WriteLine(
                        $"{frameInPath},{position.x},{position.y},{position.z},{eulerRotation.x},{eulerRotation.y},{eulerRotation.z}"
                    );

                    // Create subfolder for this time step
                    string timeStepFolder = Path.Combine(iterationFolder, $"TimeStep_{frameInPath}");
                    if (!Directory.Exists(timeStepFolder))
                        Directory.CreateDirectory(timeStepFolder);

                    // Capture 6 faces
                    CaptureAndSaveFaces(timeStepFolder, segmentRotation);
                }
            }
        }
    }

    void CaptureAndSaveFaces(string timeStepFolder, Quaternion cameraRotation)
    {
        Vector3[] localDirections = new Vector3[]
        {
            Vector3.back,    // Back
            Vector3.left,    // Left
            Vector3.forward, // Front
            Vector3.right,   // Right
            Vector3.up,      // Top
            Vector3.down     // Bottom
        };

        Vector3[] upVectors = new Vector3[]
        {
            Vector3.up,       // Back face up vector
            Vector3.up,       // Left face up vector
            Vector3.up,       // Front face up vector
            Vector3.up,       // Right face up vector
            Vector3.forward,  // Top face up
            Vector3.back      // Bottom face up
        };

        string[] faceNames = new string[]
        {
            "back", "left", "front", "right", "top", "bottom"
        };

        for (int i = 0; i < localDirections.Length; i++)
        {
            Vector3 worldDirection = cameraRotation * localDirections[i];
            Vector3 worldUp = cameraRotation * upVectors[i];
            transform.rotation = Quaternion.LookRotation(worldDirection, worldUp);

            SaveImage(timeStepFolder, faceNames[i]);
        }
    }

    void SaveImage(string timeStepFolder, string faceName)
    {
        if (Camera.main == null)
            return;

        Camera.main.targetTexture = rt;
        Camera.main.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, outputWidth, outputHeight), 0, 0);
        screenShot.Apply();

        Camera.main.targetTexture = null;
        RenderTexture.active = null;

        string filename = Path.Combine(timeStepFolder, $"{faceName}.png");
        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(filename, bytes);
    }

    void OnDestroy()
    {
        // Release resources
        if (rt != null)
        {
            rt.Release();
            Destroy(rt);
        }
        if (screenShot != null)
        {
            Destroy(screenShot);
        }
    }

    struct PathSegment
    {
        public float Distance;
        public float Rotation;
    }
}