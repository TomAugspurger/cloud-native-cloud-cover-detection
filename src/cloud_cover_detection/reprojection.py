import affine
import pyproj
import pystac
import planetary_computer
import rasterio
import rasterio.warp
import numpy as np


# Read https://r.geocompx.org/reproj-geo-data
# Look into xesfm and curvilinear grids


def reproject_to_wgs84(
    data: np.ndarray,
    transform: affine.Affine,
    src_crs: pyproj.crs.CRS,
    target_resolution=0.001,
):
    """
    Reproject a 2D array from source projection to EPSG:4326 (WGS84) using NumPy operations.

    Args:
        data: The 2D array of luminance values in the source projection
        transform: rasterio affine transform from the source data
        src_crs: The source coordinate reference system
        target_resolution: Resolution of the target grid in degrees

    Returns:
        tuple: (reprojected_data, new_transform) where reprojected_data is a 2D numpy array
    """
    height, width = data.shape
    (src_r, src_b) = transform * (width, height)
    src_bounds = (
        transform.c,
        src_b,
        src_r,
        transform.f,
    )

    # dunno if this is right
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, "EPSG:4326", width, height, *src_bounds
    )

    # Create arrays of x and y indices for the source grid
    rows, cols = np.indices((height, width))  # this seems like a lot of memory, no?

    # Convert pixel indices to source coordinates using the affine transform
    src_x = transform.c + cols * transform.a + rows * transform.b
    src_y = transform.f + cols * transform.d + rows * transform.e

    # Create the transformer from source CRS to WGS84
    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326")

    # Transform coordinates from source CRS to WGS84
    lon, lat = transformer.transform(src_x.flatten(), src_y.flatten())
    lon = lon.reshape(src_x.shape)
    lat = lat.reshape(src_y.shape)

    # Determine bounds of the reprojected grid
    min_lon, max_lon = np.min(lon), np.max(lon)
    min_lat, max_lat = np.min(lat), np.max(lat)

    # target_resolution = 0.001

    # Create regular grid in WGS84
    # target_lon = np.arange(min_lon, max_lon, target_resolution)
    # target_lat = np.arange(min_lat, max_lat, target_resolution)

    target_lon = np.linspace(min_lon, max_lon, dst_width)
    target_lat = np.linspace(min_lat, max_lat, dst_height)

    # Create regular grid in WGS84
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)

    # Create the inverse transformer (WGS84 to source CRS)
    inverse_transformer = pyproj.Transformer.from_crs("EPSG:4326", src_crs)

    # Transform target grid coordinates to source CRS
    target_x, target_y = inverse_transformer.transform(
        target_lon_grid.flatten(), target_lat_grid.flatten()
    )
    target_x = target_x.reshape(target_lon_grid.shape)
    target_y = target_y.reshape(target_lat_grid.shape)

    # Convert source coordinates to pixel indices
    inv_transform = ~transform
    pixel_x = (target_x - inv_transform.c) / inv_transform.a
    pixel_y = (target_y - inv_transform.f) / inv_transform.e

    # Interpolate using bilinear interpolation
    x0 = np.floor(pixel_x).astype(int)
    y0 = np.floor(pixel_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Ensure indices are within bounds
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    # Calculate weights for bilinear interpolation
    wx = pixel_x - x0
    wy = pixel_y - y0

    # Gather the four nearest pixels
    v00 = data[y0, x0]
    v01 = data[y0, x1]
    v10 = data[y1, x0]
    v11 = data[y1, x1]

    # Calculate the weighted average (bilinear interpolation)
    reprojected = (
        v00 * (1 - wx) * (1 - wy)
        + v01 * wx * (1 - wy)
        + v10 * (1 - wx) * wy
        + v11 * wx * wy
    )

    # Create the new transform for the reprojected data
    new_transform = rasterio.transform.from_bounds(
        min_lon, min_lat, max_lon, max_lat, len(target_lon), len(target_lat)
    )

    return reprojected, new_transform


def fetch_test_item():
    import json

    item = planetary_computer.sign(
        pystac.read_file(
            "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20250331T165849_R069_T15TVG_20250331T210313"
        )
    )
    with rasterio.open(item.assets["B02"].href) as src:
        data = src.read(1)
        with rasterio.open(
            "/datasets/toaugspurger/s2-test.tif",
            "w",
            driver="GTiff",
            count=1,
            dtype=data.dtype,
            crs=item.properties["proj:code"],
            transform=item.assets["B02"].extra_fields["proj:transform"],
            height=data.shape[0],
            width=data.shape[1],
        ) as dst:
            dst.write(data, 1)
    with open("/datasets/toaugspurger/s2-test.json", "w") as f:
        json.dump(item.to_dict(), f)


def main():
    item = pystac.read_file("/datasets/toaugspurger/s2-test.json")
    with rasterio.open("/datasets/toaugspurger/s2-test.tif") as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    # # subset for testing
    # small = data[:100, :100]

    (expected_dst, dst_transform) = rasterio.warp.reproject(
        data,
        src_transform=transform,
        src_crs=crs,
        dst_crs="EPSG:4326",
    )

    reprojected_data, new_transform = reproject_to_wgs84(data, transform, crs)

    src = rasterio.open(item.assets["B02"].href)

    # (10980, 10980)
    with rasterio.open(item.assets["B02"].href) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

        # Reproject the data to WGS84
        reprojected_data, new_transform = reproject_to_wgs84(data, transform, crs)

        # Optional: Save the reprojected data
        # with rasterio.open(
        #     'reprojected.tif', 'w',
        #     driver='GTiff',
        #     height=reprojected_data.shape[0],
        #     width=reprojected_data.shape[1],
        #     count=1,
        #     dtype=reprojected_data.dtype,
        #     crs='EPSG:4326',
        #     transform=new_transform,
        # ) as dst:
        #     dst.write(reprojected_data, 1)


if __name__ == "__main__":
    main()
