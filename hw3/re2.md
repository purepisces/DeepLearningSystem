```mermaid
graph TD
    A[Initialize ImageSimChannel] --> B[Load Embeddings from File]
    B --> C[Create FAISS Index]
    C --> D[Add Embeddings to FAISS Index]

    subgraph Update Data
        E[Update Image List from Log] --> F[Update Interacted Set]
    end

    D --> G[Call: Generate Recommendations]

    subgraph Recommendation Process
        G --> H[Get Recommendations for Images]
        H --> I[Filter Out Interacted and Recommended Items]
        I --> J[Check if Recommendation List is Sufficient]
        J --> K{Is List Sufficient?}
        K -- Yes --> L[Shuffle Recommendations]
        K -- No --> M[Increase Recommendation Count and Repeat]
    end

    L --> N[Return Final Recommendation List]
    N --> O[Return Image Names]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style G fill:#f99,stroke:#333,stroke-width:2px;
    style L fill:#ff9,stroke:#333,stroke-width:2px;
    style N fill:#f99,stroke:#333,stroke-width:2px;
