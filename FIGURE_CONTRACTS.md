# Figure contracts and visual specification

## Global specification

- Backend: Python/Matplotlib; data are unchanged from the verified CSV/analysis functions.
- Target: compact, Nature-inspired evidence figures suitable for a two-column journal.
- Colour: colour-vision-safe navy/orange/teal/purple; line style and marker shape provide redundant encoding.
- Typography: Arial/Helvetica fallback, 7.3–8.5 pt at final export size.
- Outputs: editable PDF/SVG, 400 dpi PNG, 600 dpi LZW TIFF, and grayscale preview.

## Figure contracts

1. **Pole audit** — Finding: the implemented gains increasingly miss the requested pole placement as normalized bandwidth grows. Evidence: target/exact/implemented spectral radii and the analytic coefficient defect. Visual test: the rebound and the 25% error at unity must be visible without reading the caption.
2. **Sampling convergence** — Finding: the corrected log-uniform estimator is unbiased within Monte Carlo uncertainty and its dispersion follows the expected square-root law. Evidence: signed-error 95% intervals plus standard deviation/MAE scaling. Visual test: zero crossing and the fitted slope must be readable.
3. **Noise amplification** — Finding: exact pole placement reduces both RMS and bounded-noise gains across the tested range. Evidence: absolute curves and implemented/exact ratios. Visual test: ratio remains above one and the high-bandwidth advantage is explicit.
4. **JONSWAP tuning** — Finding: the illustrative criterion has interior optima for higher sea states and selects lower bandwidth as measurement noise increases. Evidence: objective curves, all-state selections with boundary coding, and SS3 sensitivity. Visual test: boundary solutions cannot be mistaken for interior optima.
