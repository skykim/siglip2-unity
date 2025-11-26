using System.Collections.Generic;

[System.Serializable]
public class EmbeddingData
{
    public string imageName;
    public float[] embedding;
}

[System.Serializable]
public class EmbeddingIndex
{
    public List<EmbeddingData> indexList;
}

public struct SimilarityResult
{
    public string imageName;
    public float score;
}