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
    public string outputDirectory = ""; // Output directory

    public float navigationDistance = 100f; // Total distance (n) for the camera path
    public int rotationTimes = 4;           // Number of segments (k) in the path

    public float minX, maxX, minY, maxY, minZ, maxZ; // Boundaries for random position sampling
    public int maxSamplingAttempts = 100;           // Maximum attempts to find a valid path
    public int fixedLastSegmentFrameNum = 50;     // Fixed length for the last segment
    
    private int currentFrame = 0;          // Current frame counter


    void Start()
    {
        for (int i = 0; i < dataNum; i++)
        {
            // Create a new folder for this iteration
            string iterationFolder = Path.Combine(outputDirectory, $"{navigationDistance}_{rotationTimes}", $"Iteration_{i}");
            if (Directory.Exists(iterationFolder))
            {
                // Skip if folder exists
                continue;
            }
            Directory.CreateDirectory(iterationFolder);

            // Sample a random start position
            Vector3 startPosition = SampleRandomStartPosition();

            // Try to generate a valid path
            PathSegment[] path = null;
            int attempts = 0;
            bool validPath = false;
            while (validPath==false && attempts < maxSamplingAttempts)
            {
                attempts++;
                int numSteps = Mathf.RoundToInt((float)(navigationDistance / frameStep));
                path = GenerateVectorsPrecise(numSteps, rotationTimes, frameStep, fixedLastSegmentFrameNum);
                validPath = true;
                

                // Check if each segment is valid
                Vector3 currentPosition = startPosition;
                for (int j = 0; j < path.Length; j++)
                {
                    Vector3 movementDirection = Quaternion.Euler(0, path[j].Rotation, 0) * Vector3.forward;
                    Vector3 targetPosition = currentPosition + movementDirection * path[j].Distance;

                    if (!IsPathSegmentValid(currentPosition, targetPosition))
                    {
                        validPath = false;
                        break;
                    }

                    currentPosition = targetPosition;
                }
                
            }

            if (validPath == false)
            {
                Debug.LogError($"Failed to generate a valid path after {maxSamplingAttempts} attempts for iteration {i}.");
                continue;
            }

            // Perform the movement and capture images for this iteration
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
    float fixedLastSegmentFrameNum,
    int? maxDistance = null,
    double learningRate = 0.005,
    int iterations = 10000,
    double maxError = 1e-5)
    {
        System.Random random = new System.Random();
        while (true)
        {
            // Ensure the sum of distances is correct
            int adjustedN = Mathf.RoundToInt(n - fixedLastSegmentFrameNum / frameStep);
            if (adjustedN <= 0 || k < 2)
            {
                Debug.LogError("Invalid parameters: Ensure that the last segment length is smaller than total distance.");
                return null;
            }

            Debug.Log($"n = {adjustedN}, k = {k}");
            
            // Generate random integer magnitudes for the first k-1 segments
            int[] magnitudes = Enumerable.Repeat(1, k - 1).ToArray();

            if (maxDistance == null)
                maxDistance = adjustedN / 2;

            int remainingDistance = adjustedN - (k - 1);

            while (remainingDistance > 0)
            {
                int index = random.Next(0, k - 1);
                if (magnitudes[index] < maxDistance)
                {
                    magnitudes[index]++;
                    remainingDistance--;
                }
            }

            // Normalize magnitudes
            double[] normalizedMagnitudes = magnitudes.Select(m => (double)m / adjustedN).ToArray();
            double[] angles = Enumerable.Range(0, k - 1).Select(_ => random.NextDouble() * 2 * Math.PI).ToArray();

            for (int iteration = 0; iteration < iterations; iteration++)
            {
                double xSum = normalizedMagnitudes.Select((mag, i) => mag * Math.Cos(angles[i])).Sum();
                double ySum = normalizedMagnitudes.Select((mag, i) => mag * Math.Sin(angles[i])).Sum();

                if (Math.Abs(xSum) <= maxError && Math.Abs(ySum) <= maxError)
                    break;

                double[] gradX = normalizedMagnitudes.Select((mag, i) => -mag * Math.Sin(angles[i])).ToArray();
                double[] gradY = normalizedMagnitudes.Select((mag, i) => mag * Math.Cos(angles[i])).ToArray();

                for (int i = 0; i < k - 1; i++)
                {
                    angles[i] -= learningRate * adjustedN * (xSum * gradX[i] + ySum * gradY[i]);
                }
            }

            double finalXSum = normalizedMagnitudes.Select((mag, i) => mag * Math.Cos(angles[i])).Sum();
            double finalYSum = normalizedMagnitudes.Select((mag, i) => mag * Math.Sin(angles[i])).Sum();

            if (Math.Abs(finalXSum) <= maxError && Math.Abs(finalYSum) <= maxError)
            {
                PathSegment[] pathSegments = new PathSegment[k];

                for (int i = 0; i < k - 1; i++)
                {
                    float previousAngle = i == 0 ? 0 : (float)(angles[i - 1] * 180 / Math.PI);
                    float currentAngle = (float)(angles[i] * 180 / Math.PI);
                    pathSegments[i] = new PathSegment
                    {
                        Distance = (float)magnitudes[i] * frameStep,
                        Rotation = Mathf.DeltaAngle(previousAngle, currentAngle)
                    };
                    Debug.Log($"Segment {i}: Distance = {pathSegments[i].Distance}, Rotation = {pathSegments[i].Rotation}");
                }

                // Set the last segment to the fixed length
                pathSegments[k - 1] = new PathSegment
                {
                    Distance = fixedLastSegmentFrameNum * frameStep,
                    Rotation = 0 // No additional rotation, keeping the last direction
                };

                Debug.Log($"Last Segment: Distance = {pathSegments[k - 1].Distance}, Rotation = {pathSegments[k - 1].Rotation}");
                return pathSegments;
            }
            else
            {
                Debug.Log("Retrying...");
            }
        }
    }


    bool IsPathSegmentValid(Vector3 startPosition, Vector3 endPosition)
    {
        Vector3 direction = (endPosition - startPosition).normalized;
        float distance = Vector3.Distance(startPosition, endPosition);
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

            float cumulativeAngle = 0; // Track total rotation
            float frame_in_path = 0;
            foreach (var segment in path)
            {
                // Increment cumulative rotation for this segment
                cumulativeAngle += segment.Rotation;

                // Set absolute rotation for the segment
                Quaternion segmentRotation = Quaternion.Euler(0, cumulativeAngle, 0);
                transform.rotation = segmentRotation; // Set rotation once at the beginning of the segment

                // Calculate movement direction
                Vector3 movementDirection = transform.forward;
                Vector3 targetPosition = transform.position + movementDirection * segment.Distance;

                // log segment.distance and movementDirection
                int segmentFrames = Mathf.CeilToInt(segment.Distance / frameStep);
                
                for (currentFrame = 0; currentFrame < segmentFrames; currentFrame++)
                {
                    frame_in_path += 1;
                    // Interpolate position for smooth movement
                    transform.position = Vector3.Lerp(transform.position, targetPosition, (float)currentFrame / (float)segmentFrames);

                    // Log pose information (rotation remains constant for the segment)
                    Vector3 position = transform.position;
                    Vector3 eulerRotation = segmentRotation.eulerAngles; // Use the constant rotation for the segment

                    poseWriter.WriteLine($"{frame_in_path},{position.x},{position.y},{position.z},{eulerRotation.x},{eulerRotation.y},{eulerRotation.z}");

                    // Create folder for the current time step
                    string timeStepFolder = Path.Combine(iterationFolder, $"TimeStep_{frame_in_path}");
                    if (!Directory.Exists(timeStepFolder))
                    {
                        Directory.CreateDirectory(timeStepFolder);
                    }

                    // Capture images with face orientation adapted to the segment's constant rotation
                    CaptureAndSaveFaces(timeStepFolder, segmentRotation);
                }
            }
        }
    }

    void CaptureAndSaveFaces(string timeStepFolder, Quaternion cameraRotation)
    {
        Vector3[] localDirections = new Vector3[]
        {
            Vector3.back,   // Back face
            Vector3.left,   // Left face
            Vector3.forward,// Front face
            Vector3.right,  // Right face
            Vector3.up,     // Top face
            Vector3.down    // Bottom face
        };

        Vector3[] upVectors = new Vector3[]
        {
            Vector3.up,      // Back face up vector
            Vector3.up,      // Left face up vector
            Vector3.up,      // Front face up vector
            Vector3.up,      // Right face up vector
            Vector3.forward, // Top face up vector (ensure consistency for top-down faces)
            Vector3.back     // Bottom face up vector
        };

        string[] faceNames = new string[]
        {
            "back", "left", "front", "right", "top", "bottom"
        };

        for (int i = 0; i < localDirections.Length; i++)
        {
            // Calculate the world direction based on the camera's current rotation
            Vector3 worldDirection = cameraRotation * localDirections[i];
            Vector3 worldUp = cameraRotation * upVectors[i];
            transform.rotation = Quaternion.LookRotation(worldDirection, worldUp);

            // Save the image for this face
            SaveImage(timeStepFolder, faceNames[i]);
        }
    }


    void SaveImage(string timeStepFolder, string faceName)
    {
        RenderTexture rt = new RenderTexture(outputWidth, outputHeight, 24);
        Camera.main.targetTexture = rt;
        Texture2D screenShot = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);
        Camera.main.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, outputWidth, outputHeight), 0, 0);
        screenShot.Apply();
        Camera.main.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        string filename = Path.Combine(timeStepFolder, $"{faceName}.png");
        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(filename, bytes);
    }

    struct PathSegment
    {
        public float Distance; // Distance for the segment
        public float Rotation; // Rotation angle in degrees
    }
}

