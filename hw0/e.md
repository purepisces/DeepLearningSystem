```mermaid
flowchart TD
    A[Start: Load Metadata] --> B[Compute Tag and Type Totals]
    B --> C[Process User Interaction Data]
    C --> D[Calculate Tag Click Rates]
    C --> E[Calculate Type Click Rates]
    D --> F[Sort Tags by Click Rates and Recent Interaction]
    E --> G[Sort Types by Click Rates and Recent Interaction]
    F --> H[Group Artworks by Relevant Tags and Types]
    G --> H
    H --> I[Filter Out Interacted and Previously Recommended Artworks]
    I --> J[Ensure at Least One Artwork per Type]
    J --> K[Rank Remaining Artworks by Blended Scores]
    K --> L[Select Final Artworks Based on Ranking]
    L --> M[Return Selected Artworks, Tags, and Types]
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style M fill:#0f9,stroke:#333,stroke-width:2px;
    style L fill:#f96,stroke:#333,stroke-width:2px;
