using UnityEngine;
using System.IO;
using System;
using System.Collections.Generic;

/// <summary>
/// Generates Catmull-Rom splines from random start S->W->T, samples frames, 
/// captures images facing 6 directions, and checks collision via a capsule cast.
/// </summary>
public class CatmullRomTrajectorySampler : MonoBehaviour
{
    [Header("Basic Path Settings")]
    [Tooltip("Desired distance from start (S) to target (T).")]
    public float pathLength = 20f;

    [Tooltip("Minimum random offset from line S->T when placing waypoint W.")]
    public float deviationMin = 1f;

    [Tooltip("Maximum random offset from line S->T when placing waypoint W.")]
    public float deviationMax = 3f;

    [Tooltip("How many frames to sample along each Catmull-Rom segment.")]
    public int framesPerSegment = 25;

    [Header("Bounding Box for Start Position")]
    public float minX = -60f;
    public float maxX = 50f;
    public float minY = 1.6f;
    public float maxY = 1.8f;
    public float minZ = -50f;
    public float maxZ = 50f;

    [Header("Collision Checking")]
    [Tooltip("Radius used when checking for collisions along the path.")]
    public float pathCollisionRadius = 0.5f;

    [Header("Number of Episodes & Attempts")]
    [Tooltip("How many episodes (trajectories) to generate.")]
    public int dataNum = 10;

    [Tooltip("Max attempts to find a valid path before giving up on an episode.")]
    public int maxSamplingAttempts = 100;

    [Header("Output Settings")]
    [Tooltip("Output image width in pixels.")]
    public int outputWidth = 500;

    [Tooltip("Output image height in pixels.")]
    public int outputHeight = 500;

    [Tooltip("Folder path where the data will be saved.")]
    public string outputDirectory = "";

    // Internally used counters
    private float globalFrameCounter;

    // Reusable RenderTexture/Texture2D to avoid frequent allocations
    private RenderTexture renderTexture;
    private Texture2D screenShot;

    void Start()
    {
        // Allocate a single RenderTexture and Texture2D upfront
        renderTexture = new RenderTexture(outputWidth, outputHeight, 24);
        screenShot = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);

        // Generate dataNum episodes
        for (int i = 0; i < dataNum; i++)
        {
            string episodeFolder = Path.Combine(outputDirectory, $"episode_{i:D3}");
            if (Directory.Exists(episodeFolder))
            {
                // If the folder already exists, skip to avoid overwriting
                continue;
            }
            Directory.CreateDirectory(episodeFolder);

            bool validPathFound = false;
            int attempts = 0;

            while (!validPathFound && attempts < maxSamplingAttempts)
            {
                // 1) Pick a random start position S within bounding box
                Vector3 S = SampleRandomPosition();

                // 2) Attempt to build and capture a single Catmull-Rom path (S->W->T)
                if (BuildAndCaptureCatmullRom(episodeFolder, S))
                {
                    validPathFound = true;
                }
                attempts++;
            }

            if (!validPathFound)
            {
                Debug.LogWarning(
                    $"Failed to build a valid Catmull-Rom path after {maxSamplingAttempts} attempts for episode {i}."
                );
            }
            else
            {
                // Optionally force garbage collection
                GC.Collect();
            }
        }
    }

    void OnDisable()
    {
        // Cleanup
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
    /// Builds a Catmull-Rom spline that starts at S, ends at T 
    /// (which is pathLength away in a random XZ direction), 
    /// with one random waypoint W in between for a smooth "curve."
    /// Checks if the path is collision-free. If valid, samples and saves frames.
    /// Returns true if successful, false otherwise.
    /// </summary>
    bool BuildAndCaptureCatmullRom(string episodeFolder, Vector3 S)
    {
        // 1) Pick T = S + (random direction * pathLength)
        Vector3 directionXZ = SampleRandomDirectionXZ();
        Vector3 T = S + directionXZ.normalized * pathLength;

        // 2) Pick one random waypoint W that deviates from S->T
        float frac = UnityEngine.Random.Range(0.2f, 0.8f);
        Vector3 linePoint = Vector3.Lerp(S, T, frac);

        Vector3 stDir = (T - S).normalized;
        // A perpendicular in XZ is basically rotate stDir by +90 or -90 around Y
        Vector3 perpXZ = new Vector3(-stDir.z, 0f, stDir.x);

        float offset = UnityEngine.Random.Range(deviationMin, deviationMax);
        Vector3 W = linePoint + perpXZ * offset;

        // 3) Define Catmull-Rom points: p1=S, p2=W, p3=T, plus "ghost" p0, p4 for tangents
        Vector3 p1 = S;
        Vector3 p2 = W;
        Vector3 p3 = T;

        // Ghost points for Catmull-Rom
        Vector3 p0 = p1 + (p1 - p2); // reflection for tangent
        Vector3 p4 = p3 + (p3 - p2);

        // ---------------------------------------------------------------------
        // COLLISION CHECK:
        // We'll sample the entire spline at fine intervals and do a capsule cast
        // on each small segment to ensure no collisions.
        // ---------------------------------------------------------------------
        List<Vector3> sampledPoints = new List<Vector3>();

        // Sample segment A (p0,p1,p2,p3)
        for (int f = 0; f < framesPerSegment; f++)
        {
            float t = (framesPerSegment <= 1) ? 1f : (float)f / (framesPerSegment - 1);
            Vector3 catmullPos = CatmullRom(p0, p1, p2, p3, t);
            sampledPoints.Add(catmullPos);
        }
        // Sample segment B (p1,p2,p3,p4)
        for (int f = 0; f < framesPerSegment; f++)
        {
            float t = (framesPerSegment <= 1) ? 1f : (float)f / (framesPerSegment - 1);
            Vector3 catmullPos = CatmullRom(p1, p2, p3, p4, t);
            sampledPoints.Add(catmullPos);
        }

        // If any segment is invalid, we discard this path
        if (!IsFullPathValid(sampledPoints, pathCollisionRadius))
        {
            return false;
        }

        // ---------------------------------------------------------------------
        // If the path is valid, do the actual recording and capturing
        // ---------------------------------------------------------------------
        string poseFilePath = Path.Combine(episodeFolder, "camera_poses.txt");
        using (StreamWriter poseWriter = new StreamWriter(poseFilePath))
        {
            poseWriter.WriteLine("Frame,Position.x,Position.y,Position.z,Rotation.x,Rotation.y,Rotation.z");
            globalFrameCounter = 0;

            // Reset camera to S, look at W
            transform.position = p1;
            Vector3 initDir = (p2 - p1).normalized;
            transform.rotation = Quaternion.LookRotation(initDir, Vector3.up);

            // Segment A: p0,p1,p2,p3
            for (int f = 0; f < framesPerSegment; f++)
            {
                float t = (framesPerSegment <= 1) ? 1f : (float)f / (framesPerSegment - 1);
                Vector3 catmullPos = CatmullRom(p0, p1, p2, p3, t);
                transform.position = catmullPos;

                // Forward direction
                float tAhead = Mathf.Clamp01(t + 0.01f);
                Vector3 aheadPos = CatmullRom(p0, p1, p2, p3, tAhead);
                Vector3 forward = (aheadPos - catmullPos).normalized;
                if (forward.sqrMagnitude > 1e-8f)
                {
                    transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
                }

                WritePose(poseWriter);
                string timeStepFolder = CreateFrameFolder(episodeFolder, (int)globalFrameCounter);
                CaptureAndSaveFaces(timeStepFolder, transform.rotation);
                globalFrameCounter++;
            }

            // Segment B: p1,p2,p3,p4
            for (int f = 0; f < framesPerSegment; f++)
            {
                float t = (framesPerSegment <= 1) ? 1f : (float)f / (framesPerSegment - 1);
                Vector3 catmullPos = CatmullRom(p1, p2, p3, p4, t);
                transform.position = catmullPos;

                // Forward direction
                float tAhead = Mathf.Clamp01(t + 0.01f);
                Vector3 aheadPos = CatmullRom(p1, p2, p3, p4, tAhead);
                Vector3 forward = (aheadPos - catmullPos).normalized;
                if (forward.sqrMagnitude > 1e-8f)
                {
                    transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
                }

                WritePose(poseWriter);
                string timeStepFolder2 = CreateFrameFolder(episodeFolder, (int)globalFrameCounter);
                CaptureAndSaveFaces(timeStepFolder2, transform.rotation);
                globalFrameCounter++;
            }
        }

        // If we get here, we've successfully generated & captured a path
        return true;
    }

    /// <summary>
    /// Subdivides the path into segments between consecutive points in sampledPoints 
    /// and verifies each segment is collision-free with a capsule cast.
    /// </summary>
    bool IsFullPathValid(List<Vector3> sampledPoints, float radius)
    {
        for (int i = 0; i < sampledPoints.Count - 1; i++)
        {
            if (!IsPathSegmentValid(sampledPoints[i], sampledPoints[i + 1], radius))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Checks a single path segment between startPosition and endPosition 
    /// by capsule-casting for collisions.
    /// </summary>
    bool IsPathSegmentValid(Vector3 startPosition, Vector3 endPosition, float radius)
    {
        Vector3 direction = (endPosition - startPosition).normalized;
        float distance = Vector3.Distance(startPosition, endPosition);

        // Adjust the capsule's start/end to keep them near "the body" 
        // (e.g., if your agent is 2 meters tall, you can offset up by 1 meter).
        // For simplicity, let's do a short capsule from foot to slight above foot:
        Vector3 capsuleStart = startPosition + Vector3.up * radius; 
        Vector3 capsuleEnd   = endPosition   + Vector3.up * radius;

        // If there's a hit, the path is invalid
        if (Physics.CapsuleCast(capsuleStart, capsuleEnd, radius, direction, distance))
        {
            return false;
        }

        // Optional: also check bounding box constraints or other conditions here.
        if (!IsWithinBounds(startPosition) || !IsWithinBounds(endPosition))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Checks if a point is inside the user-defined bounding box.
    /// </summary>
    bool IsWithinBounds(Vector3 position)
    {
        return position.x >= minX && position.x <= maxX &&
               position.y >= minY && position.y <= maxY &&
               position.z >= minZ && position.z <= maxZ;
    }

    /// <summary>
    /// Writes current camera position/rotation to the pose file.
    /// </summary>
    void WritePose(StreamWriter poseWriter)
    {
        Vector3 pos = transform.position;
        Vector3 euler = transform.rotation.eulerAngles;
        poseWriter.WriteLine($"{globalFrameCounter},{pos.x},{pos.y},{pos.z},{euler.x},{euler.y},{euler.z}");
    }

    /// <summary>
    /// Creates a folder named by the frame index (e.g. "000", "001", etc.) under episodeFolder.
    /// </summary>
    string CreateFrameFolder(string episodeFolder, int frameIndex)
    {
        string frameCounterPadded = frameIndex.ToString("D3");
        string timeStepFolder = Path.Combine(episodeFolder, frameCounterPadded);
        if (!Directory.Exists(timeStepFolder))
        {
            Directory.CreateDirectory(timeStepFolder);
        }
        return timeStepFolder;
    }

    /// <summary>
    /// Captures 6 faces: front, back, left, right, top, bottom.
    /// We restore the original rotation after capturing each face.
    /// </summary>
    void CaptureAndSaveFaces(string timeStepFolder, Quaternion orientation)
    {
        Vector3[] localDirs = {
            Vector3.forward, Vector3.back,
            Vector3.left,    Vector3.right,
            Vector3.up,      Vector3.down
        };
        Vector3[] localUps = {
            Vector3.up,      Vector3.up,
            Vector3.up,      Vector3.up,
            Vector3.forward, Vector3.back
        };
        string[] faceNames = { "front", "back", "left", "right", "top", "bottom" };

        Quaternion originalRotation = transform.rotation;
        for (int i = 0; i < 6; i++)
        {
            Vector3 worldDir = orientation * localDirs[i];
            Vector3 worldUp = orientation * localUps[i];
            transform.rotation = Quaternion.LookRotation(worldDir, worldUp);

            SaveImage(timeStepFolder, faceNames[i]);

            // Restore to the original rotation before next face
            transform.rotation = originalRotation;
        }
    }

    /// <summary>
    /// Render from Camera.main into a RenderTexture->Texture2D->PNG, then save.
    /// </summary>
    void SaveImage(string folderPath, string faceName)
    {
        // Render into our shared RenderTexture
        Camera.main.targetTexture = renderTexture;
        Camera.main.Render();

        // Copy from RenderTexture to screenShot (Texture2D)
        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(new Rect(0, 0, outputWidth, outputHeight), 0, 0);
        screenShot.Apply();

        // Cleanup references
        Camera.main.targetTexture = null;
        RenderTexture.active = null;

        // Encode to PNG and write to disk
        byte[] bytes = screenShot.EncodeToPNG();
        string filename = Path.Combine(folderPath, faceName + ".png");
        File.WriteAllBytes(filename, bytes);
    }

    /// <summary>
    /// Sample a random position in bounding box [minX,maxX] x [minY,maxY] x [minZ,maxZ].
    /// </summary>
    Vector3 SampleRandomPosition()
    {
        float x = UnityEngine.Random.Range(minX, maxX);
        float y = UnityEngine.Random.Range(minY, maxY);
        float z = UnityEngine.Random.Range(minZ, maxZ);
        return new Vector3(x, y, z);
    }

    /// <summary>
    /// Returns a random direction in the XZ plane (Y=0).
    /// </summary>
    Vector3 SampleRandomDirectionXZ()
    {
        float angle = UnityEngine.Random.Range(0f, 360f);
        return new Vector3(Mathf.Cos(angle * Mathf.Deg2Rad), 0f, Mathf.Sin(angle * Mathf.Deg2Rad));
    }

    /// <summary>
    /// Standard 4-point Catmull-Rom (uniform) interpolation for t in [0..1].
    /// p0,p1,p2,p3 are consecutive waypoints.
    /// </summary>
    Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        // Uniform Catmull-Rom spline
        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5f * (
            2f * p1 +
            (-p0 + p2) * t +
            (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2 +
            (-p0 + 3f * p1 - 3f * p2 + p3) * t3
        );
    }
}
