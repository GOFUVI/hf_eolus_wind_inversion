# Documentation Overview

This directory collects the methodological notes that accompany every stage of the HF-radar wind inversion workflow. Each subsection distils the rationale, implementation choices, and expected outputs so that researchers and engineers can navigate the pipelines without inspecting the source code. For a broader overview of the objectives and datasets underpinning the project, consult the repository’s main [README](../README.md).

- `aggregation`: Data aggregation procedures covering pivot operations, maintenance metadata attachment, and consolidation of radar and auxiliary feeds.
- `analysis`: Post-training analysis notes focused on feature-importance studies and comparative diagnostics.
- `geo_utils`: Geospatial utilities guiding STAC catalog construction, geodesic annotation, and GeoParquet finalisation.
- `HPO`: Hyperparameter optimisation workflow detailing the search strategy, evaluation surrogates, and orchestration of tuning runs.
- `inference`: Inference playbooks describing batch scoring jobs, model selection, and artefact handling.
- `partition`: Partitioning guides explaining view materialisation, dataset concatenation, and split management.
- `training`: Training references outlining baseline learning routines and fine-tuning strategies with anti-forgetting safeguards.
