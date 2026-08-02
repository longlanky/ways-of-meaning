//! Offline reverse geocoding: GPS coordinates to the nearest city.
//!
//! The dataset is a compile-time embedded GeoNames extract (see
//! `data/cities.tsv`), so indexing never touches the network and photo
//! locations never leave the machine. "Nearest city" is deliberately simple:
//! a photo taken on a mountain above Sarajevo indexes as Sarajevo, which is
//! exactly how people remember and search for their trips.

use std::sync::OnceLock;

/// One row of `data/cities.tsv`.
struct City {
    lat: f64,
    lon: f64,
    /// Pre-formatted "Name, Country".
    place: &'static str,
}

fn cities() -> &'static [City] {
    static CITIES: OnceLock<Vec<City>> = OnceLock::new();
    CITIES.get_or_init(|| {
        include_str!("../../data/cities.tsv")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let mut parts = l.split('\t');
                let lat = parts.next()?.parse().ok()?;
                let lon = parts.next()?.parse().ok()?;
                let place = parts.next()?;
                Some(City { lat, lon, place })
            })
            .collect()
    })
}

/// The city nearest to (`lat`, `lon`), as "Name, Country", or `None` for
/// coordinates off the globe.
///
/// A linear scan over ~34k entries is a few hundred microseconds — nothing
/// next to the exiftool process spawn it accompanies, so no spatial index.
pub fn nearest_place(lat: f64, lon: f64) -> Option<&'static str> {
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return None;
    }
    let lat1 = lat.to_radians();
    let mut best: Option<(&City, f64)> = None;
    for c in cities() {
        // Equirectangular approximation: exact would be haversine, but at
        // nearest-city distances the two never disagree on the winner.
        let x = (c.lon - lon).to_radians() * lat1.cos();
        let y = (c.lat - lat).to_radians();
        let d2 = x * x + y * y;
        match best {
            Some((_, d)) if d2 >= d => {}
            _ => best = Some((c, d2)),
        }
    }
    best.map(|(c, _)| c.place)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sarajevo_coordinates_resolve_to_bosnia() {
        // The spec's own example: `wom bosnia` should find geotagged photos.
        let place = nearest_place(43.8563, 18.4131).unwrap();
        assert!(place.contains("Sarajevo"), "got {place}");
        assert!(place.contains("Bosnia"), "got {place}");
    }

    #[test]
    fn mid_atlantic_still_resolves_to_somewhere() {
        // No threshold: the nearest city is always meaningful enough to index.
        assert!(nearest_place(30.0, -40.0).is_some());
    }

    #[test]
    fn out_of_range_coordinates_are_rejected() {
        assert!(nearest_place(91.0, 0.0).is_none());
        assert!(nearest_place(0.0, 181.0).is_none());
    }

    #[test]
    fn the_dataset_is_loaded() {
        assert!(cities().len() > 30_000, "got {}", cities().len());
    }
}
