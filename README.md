![UFS-logo](./notebooks/UFS-Logo-RGB-2csolidshorizontal-72dpi-min.png)

# UFS Analysis

The UFS-Analysis package provides a suite of Python tools for evaluating NOAA UFS forecasts against verification datasets like ERA5.

## Modules
This package has 4 major modules.

- **datareader**: Download and query NOAA forecast data from s3 buckets, as well as ERA5 datasets from Google Cloud
Platform (GCP), in Zarr format.

- **regridder**: Resample (temporal), Regrid (spatial), and Align (temporal) forecast and verification datasets.

- **util**: Helper routines, along with an array of highly specialized analytical functions used for generating
diagnostic notebooks. 

- **notebooks**: A collection of interactive Jupyter notebooks showing key forecast system diagnostics.  As of May 2026,
this covers ENSO Index and Teleconnections analysis for the Seasonal Forecast System (SFS), with NAO analysis notebooks
to come shortly.  More notebook sets are planned for UFS's Sub-seasonal to Seasonal (S2S) and Medium Range Weather
(MRW) applications.

## Usage
This package is designed to run on Unix-type machines only.  Package managers such as Conda, Miniforge, and Mamba are
strongly recommended for building the prerequisite Python environment.  Make use of the `environment.yml` file in this
repository for doing so.

Alternatively, all analysis notebooks can be launched on free cloud computing platforms like Binder and Google Colab.
This package is accompanied by a public-facing Github IO page which contains url hyperlinks for running analysis
notebooks right out of the box, no extra work needed.  Explore the UFS-Analysi dashboard here:
https://ufs-community.github.io/ufs-analysis/

## Testing

```
cd tests
pytest
```
