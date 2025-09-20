# Learnings from Autonomous Fine-Tuning

This document summarizes the key, high-level insights discovered during the autonomous fine-tuning runs. It is intended to be a guide for future tuning efforts and to provide a rationale for the final recommended baseline configuration.

## Recommended Baseline Configuration (as of Cycle 10)

This configuration has been validated through multiple cycles and represents a stable, well-balanced starting point for this dataset.

*   `MIN_QUALITY_SCORE = 0.57`: The most powerful lever for controlling the quality/quantity trade-off. `0.58` was too aggressive, while `0.55` was too permissive. `0.57` appears to be the optimal sweet spot.
*   `MIN_WORDS = 7`: A precise and effective pre-filter. It surgically removes short, low-substance replies from conversational files without collateral damage to monologue content.
*   `MERGE_GAP = 1.5`: Creates more granular conversational turns, increasing the total number of pairs. Its effect is highly content-dependent and benefits rapid-fire interviews the most.
*   `CONTEXTUAL_REID_THRESHOLD = 0.72`: The conservative default is validated. A more aggressive threshold (`0.60`) increased processing complexity for zero tangible benefit in the final output, proving to be a "red herring."
*   **Weights:** The validated "core + nudge" philosophy is implemented here.
    *   `WEIGHT_SIMILARITY = 1.0` (Core Metric)
    *   `WEIGHT_OVERLAP = 1.0` (Core Metric)
    *   `WEIGHT_RATIO = 0.5` (Core Metric)
    *   `WEIGHT_AVG_CONF = 0.2` (Secondary "Nudge")
    *   `WEIGHT_PUNC = 0.2` (Secondary "Nudge")
    *   `WEIGHT_DIVERSITY = 0.5` (Secondary "Nudge")
    *   `WEIGHT_INPUT_LEN = 0.1` (Secondary "Nudge")
    *   `WEIGHT_OUTPUT_LEN = 0.1` (Secondary "Nudge")

---

## Key Insights & Discoveries

### 1. The "Alchemy" of Quality Scoring

*   **Core vs. Nudge Philosophy:** The most significant discovery is the validation of a "core + nudge" approach. The score should be dominated by the three **Core Metrics** (`SIMILARITY`, `OVERLAP`, `RATIO`), which effectively filter out structurally poor content. The five other **Nudge Metrics** (`PUNC`, `DIVERSITY`, `LEN`, etc.) should only be used with small weights (`0.1`-`0.5`) to provide a gentle push towards desirable characteristics without fundamentally altering the selection.
*   **Unpredictable Interactions:** The addition of multiple weights can have non-linear and unpredictable effects (the "alchemy"). Cycle 2 showed that adding `DIVERSITY` and `OVERLAP` weights unexpectedly *increased* the pair count, highlighting the complexity of the multi-factor system.

### 2. The Power of Primary Filters

*   **`MIN_QUALITY_SCORE` is the Master Control:** This is the most powerful and sensitive lever for controlling the final dataset size and quality. Small changes to this value have a large and immediate impact. The tuning process showed that the optimal value likely lies in a very narrow band (we tested `0.50`, `0.52`, `0.55`, `0.58`, and `0.56` before settling on `0.57`).
*   **`MIN_WORDS` is a Surgical Pre-Filter:** Unlike the quality score, `MIN_WORDS` is a very precise and predictable filter. Increasing it to 7 was highly effective at removing low-substance replies *only* from the relevant conversational files, proving its value as a targeted tool.

### 3. Structural & Technical Learnings

*   **Validate Your Assumptions (The `CONTEXTUAL_REID` Null Result):** Cycle 8 was a critical learning. We proved that a more aggressive `CONTEXTUAL_REID_THRESHOLD` was a "red herring"—it created more work for the pipeline but resulted in zero net gain in high-quality pairs. This validated the importance of testing assumptions and the value of a conservative default.
*   **Content-Dependent Effects (`MERGE_GAP`):** The effect of structural parameters is not uniform across all file types. Reducing `MERGE_GAP` to `1.5s` benefited the rapid-fire "Street Interview" file significantly more than the conversational podcasts. This shows that there is no "one-size-fits-all" setting for such parameters.
*   **Robustness is a Prerequisite for Tuning (`ffmpeg`):** The `MemoryError` in Cycle 2 made it clear that pipeline stability is paramount. Without the move to a robust `ffmpeg`-based chunking strategy, we would have been unable to process the largest files, making any tuning efforts on the weights incomplete and unreliable.
