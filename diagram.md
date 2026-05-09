# Qwen3-VL MMMU Evaluation Pipeline

```mermaid
flowchart TD
    %% Entry point
    START([start: src/main.py]) --> S1

    %% Stage 1: Resume
    subgraph S1["Stage 1: Resume State Detection"]
        direction TB
        A1["load_completed_sample_ids()"]
        A2["Read results/trajectories.jsonl"]
        A3["Scan results/samples/*.json"]
        A4{"Already at MAX_SAMPLES?"}
        A1 --> A2 --> A3 --> A4
    end

    S1 --> S2

    %% Stage 2: Model Loading
    subgraph S2["Stage 2: Model Loading"]
        direction TB
        B1["get_model_adapter('qwen_local')"]
        B2["Qwen3VLForConditionalGeneration.from_pretrained()"]
        B3["AutoProcessor.from_pretrained()"]
        B4["outlines.Generator (JSON schema constrained)"]
        B1 --> B2 --> B3 --> B4
    end

    S2 --> S3

    %% Stage 3: Dataset Loading
    subgraph S3["Stage 3: Dataset Loading"]
        direction TB
        C1["MMMU/MMMU validation split"]
        C2["4 subjects: Accounting, Architecture, Art, Biology"]
        C3["Stratified sampling: 25 per subject"]
        C4["Concatenated Dataset (100 samples)"]
        C1 --> C2 --> C3 --> C4
    end

    S3 --> S4_LOOP

    %% Stage 4: Per-Sample Loop
    subgraph S4_LOOP["Stage 4: Per-Sample Evaluation Loop"]
        direction TB

        subgraph PREP["Sample Preparation"]
            D1["Extract images (up to 7)"]
            D2["Parse options string"]
            D3["Format choices (A-D)"]
            D4["Build prompt + chat template"]
            D1 --> D2 --> D3 --> D4
        end

        PREP --> CASCADE

        subgraph CASCADE["Three-Stage Extraction Cascade"]
            direction TB
            E1["Step 1: Primary Inference\nmax_new_tokens=1024"]
            E2{"Valid JSON?"}
            E3["Step 2: Retry Inference\nmax_new_tokens=2048"]
            E4{"Valid JSON?"}
            E5["Step 3: Regex Fallback\npattern matching on raw text"]
            E6{"Answer extracted?"}
            E7["SUCCESS\nPydantic ModelResponse"]
            E8["FAILURE\nextraction_succeeded=false"]

            E1 --> E2
            E2 -- yes --> E7
            E2 -- no --> E3
            E3 --> E4
            E4 -- yes --> E7
            E4 -- no --> E5
            E5 --> E6
            E6 -- yes --> E7
            E6 -- no --> E8
        end

        CASCADE --> PERSIST

        subgraph PERSIST["Persist & Cleanup"]
            F1["Save results/samples/sample_id.json"]
            F2["Append to results/trajectories.jsonl"]
            F3["Close PIL images + torch.cuda.empty_cache()"]
            F1 --> F2 --> F3
        end
    end

    S4_LOOP -- "next uncompleted sample" --> S4_LOOP
    S4_LOOP -- "all samples done" --> S5

    %% Stage 5: Finalization
    subgraph S5["Stage 5: Finalization"]
        direction TB
        G1["rebuild_jsonl_from_samples()"]
        G2["compute_summary()"]
        G3["Save results/summary.json"]
        G4["plot_accuracy_chart()"]
        G5["plot_runtime_metrics()"]
        G6["plot_fallback_analysis()"]
        G1 --> G2 --> G3
        G2 --> G4
        G2 --> G5
        G2 --> G6
    end

    %% Outputs
    S5 --> OUTPUTS

    subgraph OUTPUTS["Outputs"]
        H1["results/summary.json"]
        H2["results/accuracy_chart.png"]
        H3["results/runtime_metrics.png"]
        H4["results/fallback_analysis.png"]
    end

    %% Resume feedback loop
    S4_LOOP -. "writes to disk" .-> RESUME_FEED["results/trajectories.jsonl\nresults/samples/*.json"]
    RESUME_FEED -. "read on restart" .-> S1

    %% Styling
    style START fill:#4a9,stroke:#333,color:#fff
    style S1 fill:#e8f0fe,stroke:#4285f4
    style S2 fill:#fce8e6,stroke:#ea4335
    style S3 fill:#e6f4ea,stroke:#34a853
    style S4_LOOP fill:#fff3e0,stroke:#fbbc04
    style S5 fill:#f3e8fd,stroke:#9334e6
    style OUTPUTS fill:#e0f7fa,stroke:#0097a7
    style RESUME_FEED fill:#f5f5f5,stroke:#999
    style E7 fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
    style E8 fill:#ffcdd2,stroke:#c62828,color:#b71c1c
```

## Key Metrics (latest run)

| Metric | Value |
|---|---|
| Model | Qwen3-VL-2B-Instruct |
| Samples | 100 (25 per subject) |
| Overall Accuracy | 41% |
| Parse Failure Rate | 0% |
| Avg Inference Time | 7.05s |
| Total Runtime | ~11.7 min |
| Best Subject | Art (56%) |
| Worst Subject | Architecture & Engineering (28%) |
