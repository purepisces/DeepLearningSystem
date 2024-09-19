flowchart TD
    A[Load Metadata] --> B[Calculate Tag and Type Counts]
    B --> C[User Interaction Data]
    C --> D[Calculate Tag Click Rates]
    C --> E[Calculate Type Click Rates]
    D --> F[Sort Tags by Click Rates and Timestamp]
    E --> G[Sort Types by Click Rates and Timestamp]
    F --> H[Group Artworks by Tag and Type]
    G --> H
    H --> I[Filter Interacted and Recommended Artworks]
    I --> J[Select One Artwork per Type]
    J --> K[Rank Remaining Artworks by Blended Score]
    K --> L[Final Artwork Selection]
    L --> M[Return Selected Artworks, Tags, and Types]
