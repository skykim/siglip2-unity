using UnityEngine;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine.UI;
using System; 
using System.Runtime.Serialization.Formatters.Binary;
using TMPro;

public class ImageSearchManager : MonoBehaviour
{
    [Header("Dependencies")]
    public ModelSigLIP2 modelSigLIP2;
    public ScrollRect searchResultsViewPort;
    public GameObject imageResultPrefab;

    [Header("Configuration")]
    public string imagesFolderPath = "Images";
    private string _embeddingFilePath;

    private EmbeddingIndex _embeddingIndex;
    private Dictionary<string, Texture2D> _loadedTextures;

    public int TOP_K = 5; 

    public TMP_Text elapsedTimeText;

    void Awake()
    {
        _embeddingFilePath = Path.Combine(Application.streamingAssetsPath, "image_embeddings.bin");
        _loadedTextures = new Dictionary<string, Texture2D>();
    }

    void Start()
    {
        LoadEmbeddingIndex();
        LoadAllTextures();
    }

    public void GenerateAndSaveEmbeddings()
    {
        if (modelSigLIP2 == null)
        {
            Debug.LogError("ModelSigLIP2 reference is missing.");
            return;
        }

        string fullImagesPath = Path.Combine(Application.streamingAssetsPath, imagesFolderPath);
        if (!Directory.Exists(fullImagesPath))
        {
            Debug.LogError($"Images directory not found at: {fullImagesPath}");
            return;
        }

        Debug.Log("Starting embedding generation...");
        _embeddingIndex = new EmbeddingIndex { indexList = new List<EmbeddingData>() };
        
        string[] imageFiles = Directory.GetFiles(fullImagesPath, "*.jpg")
                                     .Concat(Directory.GetFiles(fullImagesPath, "*.png"))
                                     .ToArray();

        int count = 0;
        
        foreach (string filePath in imageFiles)
        {
            Texture2D texture = LoadTextureFromFile(filePath);
            if (texture == null) continue;

            using (Tensor<float> embeddingTensor = modelSigLIP2.GetImageEmbedding(texture))
            {
                _embeddingIndex.indexList.Add(new EmbeddingData
                {
                    imageName = Path.GetFileName(filePath),
                    embedding = embeddingTensor.DownloadToArray()
                });
            }
            Destroy(texture);
            count++;
        }

        try
        {
            using (FileStream fs = new FileStream(_embeddingFilePath, FileMode.Create))
            using (BinaryWriter writer = new BinaryWriter(fs))
            {
                writer.Write(count);
                
                foreach (var data in _embeddingIndex.indexList)
                {
                    writer.Write(data.imageName);
                    writer.Write(data.embedding.Length); 
                    
                    foreach (float value in data.embedding)
                    {
                        writer.Write(value);
                    }
                }
            }
            Debug.Log($"Successfully generated and saved {count} embeddings to: {_embeddingFilePath} (Binary)");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to save embedding index to StreamingAssets (Binary): {e.Message}");
        }
    }

    private void LoadEmbeddingIndex()
    {
        if (File.Exists(_embeddingFilePath))
        {
            try
            {
                using (FileStream fs = new FileStream(_embeddingFilePath, FileMode.Open))
                using (BinaryReader reader = new BinaryReader(fs))
                {
                    int count = reader.ReadInt32();
                    _embeddingIndex = new EmbeddingIndex { indexList = new List<EmbeddingData>(count) };
                    
                    for (int i = 0; i < count; i++)
                    {
                        string imageName = reader.ReadString();
                        int embeddingLength = reader.ReadInt32();
                        
                        float[] embedding = new float[embeddingLength];
                        for (int j = 0; j < embeddingLength; j++)
                        {
                            embedding[j] = reader.ReadSingle();
                        }

                        _embeddingIndex.indexList.Add(new EmbeddingData
                        {
                            imageName = imageName,
                            embedding = embedding
                        });
                    }
                }
                Debug.Log($"Loaded {_embeddingIndex.indexList.Count} embeddings from binary index file in StreamingAssets.");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to load binary embedding index: {e.Message}");
                _embeddingIndex = new EmbeddingIndex { indexList = new List<EmbeddingData>() };
            }
        }
        else
        {
            Debug.LogWarning("Binary embedding index file not found in StreamingAssets. Please run 'Generate And Save Embeddings' first.");
            _embeddingIndex = new EmbeddingIndex { indexList = new List<EmbeddingData>() };
        }
    }
    
    private Texture2D LoadTextureFromFile(string filePath)
    {
        Texture2D texture = new Texture2D(2, 2);
        try
        {
            byte[] fileData = File.ReadAllBytes(filePath);
            if (texture.LoadImage(fileData))
            {
                return texture;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load texture from {filePath}: {e.Message}");
        }
        Destroy(texture);
        return null;
    }

    private void LoadAllTextures()
    {
        string fullImagesPath = Path.Combine(Application.streamingAssetsPath, imagesFolderPath);
        if (!Directory.Exists(fullImagesPath))
        {
            Debug.LogWarning($"Image folder not found for texture loading: {fullImagesPath}");
            return;
        }

        string[] imageFiles = Directory.GetFiles(fullImagesPath, "*.jpg")
                                     .Concat(Directory.GetFiles(fullImagesPath, "*.png"))
                                     .ToArray();
        
        foreach (string filePath in imageFiles)
        {
            Texture2D texture = LoadTextureFromFile(filePath);
            if (texture != null)
            {
                _loadedTextures[Path.GetFileName(filePath)] = texture;
            }
        }
    }

    public void SearchTextToImage(TMP_InputField inputField)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        string queryText = inputField.text;
        SearchTextToImage(queryText);
        stopwatch.Stop();
        elapsedTimeText.text = $"Inference Time: {stopwatch.ElapsedMilliseconds} ms ({_embeddingIndex.indexList.Count} Images)";
    }

    public void SearchTextToImage(string queryText)
    {
        if (_embeddingIndex == null || _embeddingIndex.indexList.Count == 0)
        {
            Debug.LogError("Embedding index is empty. Cannot search.");
            return;
        }

        Debug.Log($"Searching for: \"{queryText}\"");
        var results = modelSigLIP2.SearchTextToImage(queryText, _embeddingIndex);
        DisplayResults(results);
    }

    public void SearchImageToImage(Texture2D sourceImage)
    {
        if (_embeddingIndex == null || _embeddingIndex.indexList.Count == 0)
        {
            Debug.LogError("Embedding index is empty. Cannot search.");
            return;
        }
        
        Debug.Log("Searching using source image.");
        var results = modelSigLIP2.SearchImageToImage(sourceImage, _embeddingIndex);
        DisplayResults(results);
    }
    
    public float SearchTextToText(string text1, string text2)
    {
        if (modelSigLIP2 == null) return 0f;
        return modelSigLIP2.SearchTextToText(text1, text2);
    }

    private void DisplayResults(List<SimilarityResult> results)
    {
        foreach (Transform child in searchResultsViewPort.content)
        {
            Destroy(child.gameObject);
        }

        if (results == null || results.Count == 0)
        {
            Debug.Log("No relevant results found.");
            return;
        }
        
        var topResults = results.OrderByDescending(r => r.score).Take(TOP_K);

        foreach (var result in topResults)
        {
            if (_loadedTextures.TryGetValue(result.imageName, out Texture2D texture))
            {
                GameObject resultObject = Instantiate(imageResultPrefab, searchResultsViewPort.content);
                Image uiImage = resultObject.GetComponent<Image>();
                
                if (uiImage != null)
                {
                    Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.one * 0.5f);
                    uiImage.sprite = sprite;
                }
                
                Debug.Log($"Result: {result.imageName}, Score: {result.score:F4}");
            }
            else
            {
                Debug.LogWarning($"Texture not loaded for: {result.imageName}");
            }
        }
    }
}