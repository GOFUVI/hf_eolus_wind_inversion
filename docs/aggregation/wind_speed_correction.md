## Logarithmic Wind Speed Correction

The standard way to extrapolate wind speed between two heights is by using the logarithmic wind profile law (assuming a neutral atmosphere). In this case:

- **Logarithmic formula:** $$ U(y) = U(3) \frac{\ln(y/z_0)}{\ln(y_0/z_0)} $$ where $U(y_0)$ is the wind speed measured at y_0 m, $U(y)$ is the wind speed estimated at y m, and $z_0$ is the surface roughness length. For open ocean, $z_0$ is typically on the order of $10^{-4}$ m (for example, about 0.2 mm). With this typical roughness, the correction increases the wind speed by about 10–15% when moving from 3 m to 10 m (winds tend to be slightly stronger at higher elevation).

- **Alternative (power law):** Since under neutral conditions the variation is moderate, a simplified power law is sometimes used: $$ U(y) = U(y_0) \cdot \left( \frac{y}{y_0} \right)^{\alpha} $$ with an exponent $\alpha$ known as the friction (or Hellman) coefficient. A common value is $\alpha \approx 0.10$ for marine surfaces, which produces a result similar to the neutral logarithmic method.

Both approaches aim to estimate wind speed equivalent at y m. For our pipeline, we can use either the logarithmic correction (which is based on stronger physical grounds) or the power law with $\alpha=0.1$ (a widely used practical approximation). In any case, the result will be a new series of buoy wind measurements scaled to a height of y m.

### Implementation in the data-preparation pipeline

The pipeline applies the logarithmic correction through the dockerised wrapper `scripts/aggregation/apply_buoy_wind_height_correction.sh`, which invokes the Python helper `apply_buoy_wind_height_correction.py` inside a controlled container environment. The helper inspects the schema of the raw buoy table `ann_training.vilano_buoy`, regenerates the SELECT projection with the corrected `wind_speed`, and materialises the height-standardised dataset as `ann_training.vilano_buoy_height10m` (S3 prefix `ann_training/reference/vilano_buoy_height10m/`). The join step then consumes this corrected table to produce `ann_training.PIVOTS_VILANO_BUOY`, ensuring that every downstream artefact already carries 10 m wind speeds. The correction factors are parameterised (source height, target height, roughness length), allowing alternative heights or roughness assumptions without touching the orchestration logic. Sentinel fill values (≤ -9000) remain untouched so the existing validity filters continue to operate unchanged.
