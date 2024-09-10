```mermaid
graph TD
    A[Initialize ColdStartChannel] --> B[Load Metadata and Initialize Embedding Model]
    B --> C[Load or Compute Tag Embeddings]
    
    subgraph Tag Processing
        D[Extract Tags] --> E[Create Embeddings from Tags]
        E --> F[Store Tag Embeddings]
    end

    B --> G[Update User Data and Create Embeddings]
    G --> H[Generate Top-k Similar Artworks]

    subgraph Recommendation Process
        H --> I[Calculate Similarity between User and Artwork Embeddings]
        I --> J[Sort by Similarity]
        J --> K[Return Top-k Most Similar Artworks]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style B fill:#ff9,stroke:#333,stroke-width:2px;
    style H fill:#f99,stroke:#333,stroke-width:2px;
    style K fill:#ff9,stroke:#333,stroke-width:2px;
