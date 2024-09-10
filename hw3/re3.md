```mermaid
graph TD
    A[Initialize RandomRecChannel] --> B[Load Metadata]
    
    subgraph Update Data
        C[Update Interacted Set]
    end

    B --> D[Call: Generate Random Recommendations]

    subgraph Recommendation Process
        D --> E[Filter Out Interacted and Recommended Items]
        E --> F[Sample Random Items from Remaining Metadata]
        F --> G[Return Random Recommendations List]
    end

    G --> H[Return Label: Random]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style B fill:#ff9,stroke:#333,stroke-width:2px;
    style G fill:#f99,stroke:#333,stroke-width:2px;
