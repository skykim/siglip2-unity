// ImageSearchManagerEditor.cs (Must be placed in an 'Editor' folder)
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(ImageSearchManager))]
public class ImageSearchManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        ImageSearchManager manager = (ImageSearchManager)target;

        GUILayout.Space(10);
        
        // 1) Embedding Generation Button
        if (GUILayout.Button("Generate And Save Embeddings (Create Index)"))
        {
            if (EditorUtility.DisplayDialog(
                "Confirm Embedding Generation", 
                "This will calculate embeddings for ALL images in StreamingAssets/Images and overwrite the existing index file. Proceed?", 
                "Proceed", 
                "Cancel"))
            {
                manager.GenerateAndSaveEmbeddings();
            }
        }
    }
}