# cities.tsv

Nearest-place lookup data for `[extract] geo`: each line is
`latitude<TAB>longitude<TAB>City, Country`, pre-formatted for indexing.

Derived from [GeoNames](https://www.geonames.org/) `cities15000.txt` (every
city with population ≥ 15,000, ~34k rows), licensed CC-BY 4.0, with country
names from `countryInfo.txt`. GeoNames data is licensed under a Creative
Commons Attribution 4.0 License — see https://creativecommons.org/licenses/by/4.0/.

## Regenerating

```sh
curl -sO https://download.geonames.org/export/dump/cities15000.zip
curl -sO https://download.geonames.org/export/dump/countryInfo.txt
unzip -o cities15000.zip
# then run the Python snippet from this project's git history (commit that
# added this file): maps country codes to names and emits lat/lon/place rows.
```
