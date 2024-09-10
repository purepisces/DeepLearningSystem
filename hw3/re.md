```mermaid
graph TD
    A[Load Metadata from CSV] --> B[Preprocess Metadata]
    B --> C[Image Similarity Channel]
    B --> D[Common Tags Channel]
    B --> E[Same Artist Channel]
    B --> F[Random Recommendation Channel]
    
    C[Image Similarity Channel] --> C1[Compare Embeddings]
    D[Common Tags Channel] --> D1[Find Overlapping Tags]
    E[Same Artist Channel] --> E1[Match Based on Artist]
    F[Random Recommendation Channel] --> F1[Randomly Select Items]
    
    C1 --> G[Aggregate Recommended Artworks]
    D1 --> G[Aggregate Recommended Artworks]
    E1 --> G[Aggregate Recommended Artworks]
    F1 --> G[Aggregate Recommended Artworks]
    
    G --> H[Combine Results and Rank]
    H --> I[Final Output: Recommended Artworks List]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style G fill:#ff9,stroke:#333,stroke-width:2px;
    style H fill:#f99,stroke:#333,stroke-width:2px;
    style I fill:#f99,stroke:#333,stroke-width:2px;
