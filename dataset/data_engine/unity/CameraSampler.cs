using UnityEngine;
using System.IO;
using System;

public class RandomStraightPathSampler : MonoBehaviour
{
    [Header("Path Settings")]
    [Tooltip("Total length of each straight path.")]
    public float pathLength = 10f;

    [Tooltip("Distance moved between captures (frames).")]
    public float frameStep = 0.4f;

    [Tooltip("Number of paths (data) to collect.")]
    public int dataNum = 10;

    [Tooltip("Max attempts to find a valid path before giving up on an iteration.")]
    public int maxSamplingAttempts = 100;

    [Header("Bounding Box")]
    public float minX = -10f;
    public float maxX = 10f;
    public float minY = 0f;
    public float maxY = 0f;
    public float minZ = -10f;
    public float maxZ = 10f;

    [Header("Output Settings")]
    public int outputWidth = 500;     // Image width
    public int outputHeight = 500;    // Image height
    public string outputDirectory = ""; // Folder where data will be saved

    // Internally used counters
    private float frameCounter;

    // We'll allocate these once and re-use them to avoid frequent allocations
    private RenderTexture renderTexture;
    private Texture2D screenShot;

    void Start()
    {
        // Allocate a single RenderTexture and Texture2D upfront
        renderTexture = new RenderTexture(outputWidth, outputHeight, 24);
        screenShot = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);

        // Generate the specified number of paths
        for (int i = 0; i < dataNum; i++)
        {
            // Create a folder for this iteration
            string iterationFolder = Path.Combine(outputDirectory, $"episode_{i:D3}");
            if (Directory.Exists(iterationFolder))
            {
                // Skip if folder already exists
                continue;
            }
            Directory.CreateDirectory(iterationFolder);

            bool validPathFound = false;
            int attempts = 0;

            // Attempt to find a valid path up to maxSamplingAttempts times
            while (!validPathFound && attempts < maxSamplingAttempts)
            {
                // Random start position + direction on the XZ plane
                Vector3 startPosition = SampleRandomStartPosition();
                Vector3 direction = SampleRandomDirection();
                Vector3 endPosition = startPosition + direction.normalized * pathLength;

                // Check collision with a single ray
                if (IsPathValid(startPosition, endPosition))
                {
                    // If valid, capture images along the path
                    CaptureAndSaveStraightPath(iterationFolder, startPosition, endPosition);
                    validPathFound = true;
                }
                attempts++;
            }

            if (!validPathFound)
            {
                Debug.LogWarning(
                    $"Failed to find a valid straight path after {maxSamplingAttempts} attempts for iteration {i}."
                );
            }
            else
            {
                // Optional: Force a garbage collection
                GC.Collect();
            }
        }
    }

    /// <summary>
    /// Clean up the RenderTexture and Texture2D we created, to prevent leaks.
    /// </summary>
    void OnDisable()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
            renderTexture = null;
        }
        if (screenShot != null)
        {
            Destroy(screenShot);
            screenShot = null;
        }
    }

    /// <summary>
    /// Samples a random position within our bounding box.
    /// </summary>
    Vector3 SampleRandomStartPosition()
    {
        return new Vector3(
            UnityEngine.Random.Range(minX, maxX),
            UnityEngine.Random.Range(minY, maxY),
            UnityEngine.Random.Range(minZ, maxZ)
        );
    }

    /// <summary>
    /// Samples a random horizontal direction (on the XZ-plane).
    /// </summary>
    Vector3 SampleRandomDirection()
    {
        float angle = UnityEngine.Random.Range(0f, 360f);
        return new Vector3(Mathf.Sin(angle * Mathf.Deg2Rad), 0f, Mathf.Cos(angle * Mathf.Deg2Rad));
    }

    /// <summary>
    /// Checks if the straight path from start to end is free of obstacles (Raycast).
    /// </summary>
    bool IsPathValid(Vector3 startPosition, Vector3 endPosition)
    {
        Vector3 direction = (endPosition - startPosition).normalized;
        float distance = Vector3.Distance(startPosition, endPosition);
        // If we hit something, path is invalid
        return !Physics.Raycast(startPosition, direction, distance);
    }

    /// <summary>
    /// Moves from start to end in increments of frameStep, capturing 6 faces at each step.
    /// </summary>
    void CaptureAndSaveStraightPath(string iterationFolder, Vector3 startPosition, Vector3 endPosition)
    {
        // Set the initial camera position
        transform.position = startPosition;

        // The path orientation is the heading direction for the entire path
        Quaternion pathOrientation = Quaternion.LookRotation(
            (endPosition - startPosition).normalized,
            Vector3.up
        );

        // We'll record camera poses in camera_poses.txt
        string poseFilePath = Path.Combine(iterationFolder, "camera_poses.txt");
        using (StreamWriter poseWriter = new StreamWriter(poseFilePath))
        {
            poseWriter.WriteLine("Frame,Position.x,Position.y,Position.z,Rotation.x,Rotation.y,Rotation.z");

            float distance = Vector3.Distance(startPosition, endPosition);
            int totalFrames = Mathf.CeilToInt(distance / frameStep);

            frameCounter = 0;

            for (int frameIndex = 0; frameIndex < totalFrames; frameIndex++)
            {
                // Interpolate position between start and end
                float t = (float)frameIndex / (float)(totalFrames - 1);
                transform.position = Vector3.Lerp(startPosition, endPosition, t);

                // At each step, we set the camera to the path heading direction
                transform.rotation = pathOrientation;

                // Record pose
                Vector3 position = transform.position;
                Vector3 eulerRotation = transform.rotation.eulerAngles;
                poseWriter.WriteLine(
                    $"{frameCounter},{position.x},{position.y},{position.z},{eulerRotation.x},{eulerRotation.y},{eulerRotation.z}"
                );

                // Subfolder for this time step
                int frameCounterInt = (int)frameCounter;
                string frameCounterPadded = frameCounterInt.ToString("D3");


                string timeStepFolder = Path.Combine(iterationFolder, frameCounterPadded);
                if (!Directory.Exists(timeStepFolder))
                {
                    Directory.CreateDirectory(timeStepFolder);
                }

                // Capture the 6 faces (front face = heading direction)
                CaptureAndSaveFaces(timeStepFolder, pathOrientation);

                frameCounter++;
            }
        }
    }

    /// <summary>
    /// Captures 6 faces: front, back, left, right, top, bottom.
    /// 
    /// The front face is 'pathOrientation * Vector3.forward',
    /// which aligns with the heading direction of the path.
    /// We restore pathOrientation after capturing each face 
    /// so the final orientation remains the heading direction.
    /// </summary>
    void CaptureAndSaveFaces(string timeStepFolder, Quaternion pathOrientation)
    {
        // We'll define 6 directions and 6 up vectors, front face first 
        // so that front is indeed the heading direction.
        Vector3[] localDirections = new Vector3[]
        {
            Vector3.forward, // front
            Vector3.back,    // back
            Vector3.left,    // left
            Vector3.right,   // right
            Vector3.up,      // top
            Vector3.down     // bottom
        };

        Vector3[] upVectors = new Vector3[]
        {
            Vector3.up,      // front
            Vector3.up,      // back
            Vector3.up,      // left
            Vector3.up,      // right
            Vector3.forward, // top
            Vector3.back     // bottom
        };

        string[] faceNames = new string[]
        {
            "front", "back", "left", "right", "top", "bottom"
        };

        // We'll store the original rotation (heading) so we can restore it
        // after capturing each face.
        Quaternion originalRotation = transform.rotation;

        for (int i = 0; i < localDirections.Length; i++)
        {
            Vector3 worldDirection = pathOrientation * localDirections[i];
            Vector3 worldUp = pathOrientation * upVectors[i];
            transform.rotation = Quaternion.LookRotation(worldDirection, worldUp);
            SaveImage(timeStepFolder, faceNames[i]);

            // Restore orientation to the heading direction
            transform.rotation = originalRotation;
        }
    }

    /// <summary>
    /// Render the current camera view (MainCamera) into a single shared RenderTexture,
    /// then copy it to a single shared Texture2D, and save as PNG.
    /// </summary>
    void SaveImage(string folderPath, string faceName)
    {
        Camera.main.targetTexture = renderTexture;
        Camera.main.Render();

        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(new Rect(0, 0, outputWidth, outputHeight), 0, 0);
        screenShot.Apply();

        Camera.main.targetTexture = null;
        RenderTexture.active = null;

        byte[] bytes = screenShot.EncodeToPNG();
        string filename = Path.Combine(folderPath, faceName + ".png");
        File.WriteAllBytes(filename, bytes);
    }
}
