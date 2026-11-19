# Data and source notice

The archive owner explicitly authorized including these processed aggregate files in the repository update:

- `price_15min_fullyear.parquet`: timestamp index and aggregate price.
- `load_15min_fullyear.parquet`: timestamp index and aggregate load.
- `ev_demand_15min_fullyear.parquet`: timestamp index and aggregate EV demand.
- `soh_params_fullyear.json`: three numeric degradation parameters.

The files are byte-identical to the SHA-bound inputs used by this experiment. Each parquet has one numeric data column and a timestamp index; this package contains no source meter identifiers, individual-meter raw files, CVs, student identifiers or contact details. The stored files contain 560,640 rows, but this experiment repeatedly uses only the first 17,520 source rows. Stored repetition is not additional independently collected data.

External provenance and third-party redistribution/license terms for these processed aggregates remain unverified. Archive-owner authorization is recorded as authorization to include the files, not proof of underlying third-party rights. No new license is asserted for the data or the inherited project code. Existing third-party libraries retain their own licenses and are installed separately. Repository publication does not resolve these provenance or license questions.

The filename `fullyear` and older H1 labels are retained for identity only. The actual evaluation window begins 2023-01-01 and ends 2023-07-02 11:45, comprising 182 complete days and 48 remaining 15-minute rows.
