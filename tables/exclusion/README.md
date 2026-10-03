# Label-matching exclusion rates

Reconstructed flows of CIC-IDS2017 and CSE-CIC-IDS2018 are labelled by matching
them to the corrected DistriNet registries; unmatched flows are excluded
(27.93% and 37.81% of the reconstructed flows, respectively).

- `*_by_flow_length.csv`: exclusion rate by total packet count of the
  reconstructed flow.
- `*_registry_coverage_by_class.csv`: for each registry class, the number of
  registry entries and how many are matched by a reconstructed flow.
  Unmatched flows carry no label, so the per-class view is given from the
  registry side.
