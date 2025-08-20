def locstream_to_latlon(locstream: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:

    if 'lat' in locstream.dims or 'lon' in locstream.dims:
        print("locstream has lat/lon dimensions and is therefore not a locstream at all! Returning NULL")
        return None

    # Standardize coordinate labels
    locstream = Regrid.standardize_coords(locstream)

    # Rename "values" to "locations" as expected by xESMF
    locstream = locstream.rename({"values": "locations"})

    # Extract all lats and lons
    lats = locstream['lat']
    lons = locstream['lon']

    # Instantiate dataset for input to xESMF
    locs = xr.Dataset({
        'lat': (['locations'], np.ravel(lats.values)),
        'lon': (['locations'], np.ravel(lons.values))
    })

    # Construct target grid
    target_latlon_grid = xr.Dataset({'lat': np.unique(lats.values), 'lon': np.unique(lons.values)})

    # Instantiate xESMF regridder
    regridder = xe.Regridder(
        locs,
        target_latlon_grid,
        'nearest_s2d', # <-- Ensure every target grid cell has a value (but beware the imputation)
        locstream_in=True,
        reuse_weights=False
    )

    print("DEBUG: Regridder instantiated. Now running regridder.")

    # Regrid to lat-lon
     return regridder(locstream)
