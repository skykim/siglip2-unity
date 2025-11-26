using UnityEngine;
using Microsoft.ML.Tokenizers;
using System.IO;
using System.Linq;
using System;
using Unity.InferenceEngine;
using FF = Unity.InferenceEngine.Functional;
using System.Collections.Generic;

public class ModelSigLIP2 : MonoBehaviour
{
    [Header("Model Assets")]
    public ModelAsset textModelAsset;
    public ModelAsset imageModelAsset;

    private Worker _textEngine;
    private Worker _imageEngine;
    private Worker _dotScore;
    private SentencePieceTokenizer _tokenizer;

    private const BackendType BACKEND = BackendType.GPUCompute;
    private const int IMAGE_WIDTH = 224;
    private const int IMAGE_HEIGHT = 224;
    private const int CHANNELS = 3;
    private const int _features = 768;
    private const int _maxTokenLength = 64;

    void Start()
    {
        Initialize();
        RunWarmup();
    }

    void OnDestroy()
    {
        _textEngine?.Dispose();
        _imageEngine?.Dispose();
        _dotScore?.Dispose();
    }

    private void Initialize()
    {
        if (_imageEngine != null && _textEngine != null && _dotScore != null) return;
        Debug.Log("Initializing Inference Engines...");
        
        if (textModelAsset == null || imageModelAsset == null) { Debug.LogError("Model assets are not assigned in the inspector!"); return; }
        
        var tokenizerModelPath = Path.Combine(Application.streamingAssetsPath, "tokenizer.model");
        if (!File.Exists(tokenizerModelPath)) { Debug.LogError($"Tokenizer model not found at: {tokenizerModelPath}"); return; }
        using (Stream tokenizerModelStream = new FileStream(tokenizerModelPath, FileMode.Open, FileAccess.Read)) { _tokenizer = SentencePieceTokenizer.Create(tokenizerModelStream); }
        
        Model textModel = ModelLoader.Load(textModelAsset);
        Model imageModel = ModelLoader.Load(imageModelAsset);
        _textEngine = new Worker(textModel, BACKEND);
        _imageEngine = new Worker(imageModel, BACKEND);
        
        FunctionalGraph dotScoreGraph = new FunctionalGraph();
        FunctionalTensor x = dotScoreGraph.AddInput<float>(new TensorShape(1, _features));
        FunctionalTensor y = dotScoreGraph.AddInput<float>(new DynamicTensorShape(-1, _features));
        FunctionalTensor epsilon = FF.Constant(1e-8f);
        FunctionalTensor x_l2_norm = FF.Sqrt(FF.ReduceSum(x * x, 1, true));
        FunctionalTensor normalized_x = x / (x_l2_norm + epsilon);
        FunctionalTensor y_l2_norm = FF.Sqrt(FF.ReduceSum(y * y, 1, true));
        FunctionalTensor normalized_y = y / (y_l2_norm + epsilon);
        FunctionalTensor reduce = FF.ReduceSum(normalized_x * normalized_y, 1);
        Model dotScoreModel = dotScoreGraph.Compile(reduce);
        _dotScore = new Worker(dotScoreModel, BACKEND);
        
        Debug.Log("Engines Initialized Successfully.");
    }
    
    private void RunWarmup()
    {
        Debug.Log("Running Warmup...");

        try
        {
            using (var textEmb = GetTextEmbedding("warmup text")) {}

            int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS;
            float[] ones = Enumerable.Repeat(1.0f, imageSize).ToArray();
            using (var dummyImageTensor = new Tensor<float>(new TensorShape(1, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), ones))
            {
                _imageEngine.SetInput("pixel_values", dummyImageTensor);
                _imageEngine.Schedule();
                using (_imageEngine.PeekOutput().ReadbackAndClone()) {}
            }
            
            int singleFeatureSize = _features;
            int multiFeatureSize = 4 * _features;
            using (var A = new Tensor<float>(new TensorShape(1, _features), new float[singleFeatureSize]))
            using (var B = new Tensor<float>(new TensorShape(4, _features), new float[multiFeatureSize]))
            {
                _dotScore.SetInput("input_0", A);
                _dotScore.SetInput("input_1", B);
                _dotScore.Schedule();
                using (_dotScore.PeekOutput().ReadbackAndClone()) {}
            }

            Debug.Log("Warmup Complete.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Warmup failed: {e.Message}");
        }
    }

    public Tensor<float> GetImageEmbedding(Texture image)
    {
        using Tensor<float> inputTensor = TextureConverter.ToTensor(image, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS);
        _imageEngine.SetInput("pixel_values", inputTensor);
        _imageEngine.Schedule();
        var output = _imageEngine.PeekOutput() as Tensor<float>;
        return output.ReadbackAndClone();
    }
    
    public Tensor<float> GetTextEmbedding(string text)
    {
        string processedText = $"{text}";
        var tokenIds = _tokenizer.EncodeToIds(processedText, addBeginningOfSentence: false, addEndOfSentence: true).ToList();
        while (tokenIds.Count < _maxTokenLength) { tokenIds.Add(0); }
        var finalTokenIds = tokenIds.Take(_maxTokenLength).ToArray();
        var shape = new TensorShape(1, _maxTokenLength);
        using var inputIdsTensor = new Tensor<int>(shape, finalTokenIds);
        _textEngine.SetInput("input_ids", inputIdsTensor);
        _textEngine.Schedule();
        var output = _textEngine.PeekOutput() as Tensor<float>;
        return output.ReadbackAndClone();
    }
    
    private float[] GetDotScore(Tensor<float> A, Tensor<float> B)
    {
        _dotScore.SetInput("input_0", A);
        _dotScore.SetInput("input_1", B);
        _dotScore.Schedule();
        var output = _dotScore.PeekOutput() as Tensor<float>;
        using (var cpuOutput = output.ReadbackAndClone())
        {
            return cpuOutput.DownloadToArray();
        }
    }

    private Tensor<float> CombineEmbeddings(List<Tensor<float>> embeddings)
    {
        if (embeddings == null || embeddings.Count == 0) return null;

        int numEmbeddings = embeddings.Count;
        float[] combinedData = new float[numEmbeddings * _features];
        int featureSizeInBytes = _features * sizeof(float);
        
        for (int i = 0; i < numEmbeddings; i++)
        {
            float[] data;
            using (var cpuTensor = embeddings[i].ReadbackAndClone())
            {
                data = cpuTensor.DownloadToArray();
            }
            Buffer.BlockCopy(data, 0, combinedData, i * featureSizeInBytes, featureSizeInBytes);
        }

        return new Tensor<float>(new TensorShape(numEmbeddings, _features), combinedData);
    }

    public float[] Text2ImageSimilarity(string text, List<Texture> targetImages)
    {
        if (targetImages == null || targetImages.Count == 0) return new float[0];
        
        using (Tensor<float> textEmbedding = GetTextEmbedding(text))
        {
            List<Tensor<float>> imageEmbeddings = new List<Tensor<float>>();
            foreach (var image in targetImages)
            {
                imageEmbeddings.Add(GetImageEmbedding(image));
            }

            using (Tensor<float> combinedImageTensor = CombineEmbeddings(imageEmbeddings))
            {
                foreach (var emb in imageEmbeddings) emb.Dispose();

                return GetDotScore(textEmbedding, combinedImageTensor);
            }
        }
    }

    public float[] Image2ImageSimilarity(Texture sourceImage, List<Texture> targetImages)
    {
        if (targetImages == null || targetImages.Count == 0) return new float[0];

        using (Tensor<float> sourceImageEmbedding = GetImageEmbedding(sourceImage))
        {
            List<Tensor<float>> imageEmbeddings = new List<Tensor<float>>();
            foreach (var image in targetImages)
            {
                imageEmbeddings.Add(GetImageEmbedding(image));
            }

            using (Tensor<float> combinedImageTensor = CombineEmbeddings(imageEmbeddings))
            {
                foreach (var emb in imageEmbeddings) emb.Dispose();

                return GetDotScore(sourceImageEmbedding, combinedImageTensor);
            }
        }
    }

    public float[] TextSimilarity(string sourceText, List<string> targetTexts)
    {
        if (targetTexts == null || targetTexts.Count == 0) return new float[0];
        
        using (Tensor<float> sourceTextEmbedding = GetTextEmbedding(sourceText))
        {
            List<Tensor<float>> textEmbeddings = new List<Tensor<float>>();
            foreach (var text in targetTexts)
            {
                textEmbeddings.Add(GetTextEmbedding(text));
            }

            using (Tensor<float> combinedTextTensor = CombineEmbeddings(textEmbeddings))
            {
                foreach (var emb in textEmbeddings) emb.Dispose();

                return GetDotScore(sourceTextEmbedding, combinedTextTensor);
            }
        }
    }
    public List<SimilarityResult> SearchTextToImage(string queryText, EmbeddingIndex index)
    {
        if (index == null || index.indexList.Count == 0) return new List<SimilarityResult>();
        int N = index.indexList.Count;

        using (Tensor<float> textEmbedding = GetTextEmbedding(queryText))
        {
            float[] combinedEmbeddingsArray = new float[N * _features];
            int featureSizeInBytes = _features * sizeof(float);

            for (int i = 0; i < N; i++)
            {
                Buffer.BlockCopy(index.indexList[i].embedding, 0, combinedEmbeddingsArray, i * featureSizeInBytes, featureSizeInBytes);
            }

            using (Tensor<float> combinedImageTensor = new Tensor<float>(new TensorShape(N, _features), combinedEmbeddingsArray))
            {
                float[] scores = GetDotScore(textEmbedding, combinedImageTensor);

                List<SimilarityResult> results = new List<SimilarityResult>();
                for (int i = 0; i < N; i++)
                {
                    results.Add(new SimilarityResult
                    {
                        imageName = index.indexList[i].imageName,
                        score = scores[i]
                    });
                }
                return results;
            }
        }
    }

    public List<SimilarityResult> SearchImageToImage(Texture queryImage, EmbeddingIndex index)
    {
        if (index == null || index.indexList.Count == 0) return new List<SimilarityResult>();
        int N = index.indexList.Count;

        using (Tensor<float> imageEmbedding = GetImageEmbedding(queryImage))
        {
            float[] combinedEmbeddingsArray = new float[N * _features];
            int featureSizeInBytes = _features * sizeof(float);

            for (int i = 0; i < N; i++)
            {
                Buffer.BlockCopy(index.indexList[i].embedding, 0, combinedEmbeddingsArray, i * featureSizeInBytes, featureSizeInBytes);
            }

            using (Tensor<float> combinedTargetTensor = new Tensor<float>(new TensorShape(N, _features), combinedEmbeddingsArray))
            {
                float[] scores = GetDotScore(imageEmbedding, combinedTargetTensor);

                List<SimilarityResult> results = new List<SimilarityResult>();
                for (int i = 0; i < N; i++)
                {
                    results.Add(new SimilarityResult
                    {
                        imageName = index.indexList[i].imageName,
                        score = scores[i]
                    });
                }
                return results;
            }
        }
    }

    public float SearchTextToText(string text1, string text2)
    {
        float[] scores = TextSimilarity(text1, new List<string> { text2 });
        return scores.Length > 0 ? scores[0] : 0f;
    }
}