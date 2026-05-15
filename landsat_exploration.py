import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping

# Load and reproject NYC boundary shapefile
nyc = gpd.read_file('Borough-Boundaries/geo_export_23fbd979-f9f6-4bf8-afd5-1da2dcf723ea.shp')

with rasterio.open('LC08_L2SP_013032_20250611_20250618_02_T1_ST_B10.TIF') as src:
    landsat_crs = src.crs

nyc = nyc.to_crs(landsat_crs)

# Get NYC geometries for masking
nyc_geoms = [mapping(geom) for geom in nyc.geometry]

# Crop LST to NYC boundary
with rasterio.open('LC08_L2SP_013032_20250611_20250618_02_T1_ST_B10.TIF') as src:
    lst_nyc, nyc_transform = rasterio_mask(src, nyc_geoms, crop=True)
    nyc_profile = src.profile

# Convert to Celsius
lst_nyc = lst_nyc[0].astype(float)
lst_nyc = lst_nyc * 0.00341802 + 149.0 - 273.15
lst_nyc[lst_nyc < -50] = np.nan
lst_nyc[lst_nyc > 80] = np.nan

# Load QA band and reproject water mask
with rasterio.open('LC08_L2SP_013032_20250611_20250618_02_T1_QA_PIXEL.TIF') as src:
    qa_nyc, _ = rasterio_mask(src, nyc_geoms, crop=True)

qa_nyc = qa_nyc[0]
water_mask = (qa_nyc & (1 << 7)) != 0
lst_nyc[water_mask] = np.nan


from scipy import ndimage

# Compute temperature percentiles over valid land pixels only
valid_pixels = lst_nyc[~np.isnan(lst_nyc)]

# Compute cluster sizes at each threshold
for percentile in [70, 85, 95]:
    threshold = np.percentile(valid_pixels, percentile)
    mask = lst_nyc > threshold
    labeled, num = ndimage.label(mask)
    sizes = np.array([np.sum(labeled == i) for i in range(1, num + 1)])
    
    plt.figure(figsize=(8, 5))
    plt.loglog(np.sort(sizes)[::-1], np.arange(1, len(sizes) + 1), 'o', markersize=2)
    plt.xlabel('Cluster Size (pixels)')
    plt.ylabel('Rank')
    plt.title(f'Heat Islet Size Distribution — {percentile}th Percentile')
    plt.savefig(f'size_dist_{percentile}.png', dpi=150)
    plt.show()

from scipy import stats

for percentile in [70, 85, 95]:
    threshold = np.percentile(valid_pixels, percentile)
    mask = lst_nyc > threshold
    labeled, num = ndimage.label(mask)
    sizes = np.array([np.sum(labeled == i) for i in range(1, num + 1)])
    
    # Remove the giant percolating cluster (largest outlier)
    sizes = np.sort(sizes)[:-1]  
    
    # Fit power law on log-log scale
    log_sizes = np.log10(sizes)
    log_rank = np.log10(np.arange(len(sizes), 0, -1))
    
    slope, intercept, r, p, se = stats.linregress(log_sizes, log_rank)
    
    print(f'Percentile: {percentile}')
    print(f'Scaling exponent β: {-slope:.3f}')
    print(f'R²: {r**2:.3f}')
    print(f'Compare to Shreevastava: β = 1.88 ± 0.23')
    print()