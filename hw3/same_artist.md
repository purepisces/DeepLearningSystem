```mermaid
graph TD
    A[Initialize SameArtistChannel] --> B[Load Metadata]
    B --> C[Group Artworks by Artist]

    subgraph Update Data
        D[Fetch Artist List from Logs] --> E[Get Artworks for Each Artist]
        E --> F[Shuffle Artworks for Each Artist]
        F --> G[Update Candidates List]
        G --> H[Update Interacted Set]
    end

    C --> I[Call: Generate Artist Recommendations]

    subgraph Recommendation Process
        I --> J[Filter Out Interacted and Recommended Items]
        J --> K[Generate Recommendations for Each Artist]
        K --> L[Return Artist Recommendations List]
    end

    L --> M[Return Artist Names]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style B fill:#ff9,stroke:#333,stroke-width:2px;
    style C fill:#f99,stroke:#333,stroke-width:2px;
    style M fill:#f99,stroke:#333,stroke-width:2px;
