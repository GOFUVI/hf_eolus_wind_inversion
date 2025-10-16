# Station Bearing & Distance View

## Purpose
`scripts/geo_utils/add_station_bearing_distance_view.sh` creates or replaces an Athena view that augments a geometry-bearing source table with per-station great-circle metrics. For each station supplied through `--station name:lat:lon`, the view exposes `<alias>_bearing` (degrees from geographic north) and `<alias>_dist_km` (geodesic distance in kilometres) alongside the original columns. The geometry column is treated as WGS84 WKB and the view relies on Athena geospatial functions for the calculations.

## Prerequisites
- AWS CLI v2 with permissions to run Athena queries and manage Glue views.
- The source table must include a WKB geometry column (default `geometry`) describing points on the Earth's surface; WGS84 coordinates are assumed.
- Optional: `--results-s3` expects an accessible S3 prefix for Athena scratch output when the query is executed.

## Key Options
- `--source <db.table>`: Fully-qualified source table or view containing the geometry column.
- `--view <db.view>`: Target view name that will be created or replaced.
- `--station <name:lat:lon>`: Repeatable flag defining one or more reference stations. The script sanitises `name` to build the `<alias>_bearing` and `<alias>_dist_km` column names.
- `--geometry-column <name>`: Override the geometry column (defaults to `geometry`).
- `--preview`: Print the generated `CREATE OR REPLACE VIEW` statement without executing it.
- `--results-s3 <s3://...>`: Athena output bucket; required unless `--preview` is used.
- `--profile` / `--region`: AWS CLI profile and region; falls back to environment or CLI defaults.

## Computation Details
- **Point evaluation**: The script derives the reference point through `TRY(ST_Centroid(ST_GeomFromBinary(src.<geometry>)))`. While this is a reasonable proxy for compact geometries, note that Athena's `ST_Centroid` may fall outside concave polygons or highly elongated shapes; in such cases the resulting bearings and distances are computed from that off-centroid location. When the geometry is already a point, the centroid coincides with the original coordinate, so no displacement occurs. The `TRY` wrapper keeps the query resilient, propagating nulls when the centroid cannot be computed.
- **Latitude/longitude extraction**: `ST_X` and `ST_Y` convert the centroid to longitude and latitude (degrees) in CRS84, which Athena internally maps to WGS84. These values are passed as double precision floats to the trigonometric functions.
- **Distance**: The geodesic distance uses the haversine formulation [1]:
  - `a = sin²((Δφ)/2) + cos φ₁ · cos φ₂ · sin²((Δλ)/2)`
  - `c = 2 · atan2(√min(1,a), √max(0,1−a))`
  - `distance = R · c`
  The script fixes the Earth mean radius `R` to 6371.0088 km (IUGG standard derived from the Geodetic Reference System 1980), providing sub-metre agreement with Vincenty within the typical spatial extent of the HF radar grid (tens of kilometres).
- **Bearing**: Forward azimuth from the station to the observation employs the great-circle bearing equation [2]:
  - `θ = atan2( sin Δλ · cos φ₂ , cos φ₁ · sin φ₂ − sin φ₁ · cos φ₂ · cos Δλ )`
  The script wraps `θ` into `[0, 360)` degrees by adding 360 and applying `mod`. Bearings follow the navigation convention where 0° is geographic north and values increase clockwise.
- **Numerical safeguards**: The view guards against null geometries and clamps the `a` term of the haversine to `[0,1]` before the square root to avoid domain errors from floating-point noise. All trigonometric inputs are converted to radians via Athena's `radians()` helper.

## Precision Considerations
- The haversine assumption of a spherical Earth introduces sub-0.1% error compared with ellipsoidal solutions for distances under 200 km, which is acceptable for directional feature engineering. If you need centimetre-level accuracy for long baselines, replace the logic with spheroid-aware functions (e.g., `ST_Distance` on geography types once Athena supports them).
- The centroid-based point collapses complex geometries to a single location. For elongated radar cells this can shift the reference by several hundred metres. When higher fidelity is required, consider passing point geometries or adapting the script to sample multiple vertices.
- All calculations operate in double precision; rounding to four decimal places when storing bearings/distances keeps Athena query sizes manageable without losing meaningful precision.

## Typical Usage
```bash
scripts/geo_utils/add_station_bearing_distance_view.sh \
  --source analytics_db.site1_PIVOT \
  --view analytics_db.site1_PIVOT_FEATURES \
  --station site1:12.345678:-45.678912 \
  --results-s3 s3://<your-bucket>/athena-query-results/ \
  --profile your_profile \
  --region us-east-1
```
This matches the feature-enrichment step executed early in `run_ann_pipeline.sh` before any hyper-parameter optimisation workflows begin.

## Behaviour Notes
- Bearings follow the mathematical convention used in navigation: `0°` points north, and values increase clockwise.
- Distances are computed with the haversine formulation using an Earth radius of 6371.0088 km.
- Geometry centroids are evaluated via `TRY(ST_Centroid(ST_GeomFromBinary(...)))`; the centroid may lie outside concave shapes, but the `TRY` wrapper avoids hard query failures. When the geometry is a point, the centroid matches the original coordinate and no bias is introduced. Null geometries or centroid failures propagate null metrics.
- Glue databases referenced in `--view` are created on the fly when they do not exist yet.

## Troubleshooting
- *`--station` missing*: At least one station is required; repeated flags are supported for multiple stations.
- *Region not resolved*: Provide `--region` explicitly when the chosen profile lacks a default.
- *Unexpected column names*: Station aliases are derived from the slugified `name` segment; avoid characters other than letters, digits, and underscores if you need predictable column names.

## References
1. Sinnott, R. W. 1984. "Virtues of the Haversine." Sky & Telescope 68(2): 159.
2. National Geospatial-Intelligence Agency. 2019. *The American Practical Navigator (Bowditch)*, Chapter 7: Great-Circle Sailing.

## File Reference
- `scripts/geo_utils/add_station_bearing_distance_view.sh`
