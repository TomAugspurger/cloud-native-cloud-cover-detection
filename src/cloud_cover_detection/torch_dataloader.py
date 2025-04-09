import asyncio
import pathlib
import gcsfs
import xarray as xr
import structlog


structlog.configure()
logger = structlog.get_logger()


async def get_if_not_exists(
    fs: gcsfs.GCSFileSystem, rpath: str, lpath: pathlib.Path, sem: asyncio.Semaphore
) -> None:
    if not lpath.exists():
        async with sem:
            logger.info("get - begin", rpath=rpath, lpath=lpath)
            await fs._get(rpath, str(lpath))
            logger.info("get - success", rpath=rpath, lpath=lpath)
    else:
        logger.info("get - already exists", rpath=rpath, lpath=lpath)


async def clone():
    fs = gcsfs.GCSFileSystem(token="anon")
    PREFIX = (
        "weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
    )
    ds = xr.open_dataset(f"gs://{PREFIX}")
    N = 24
    MAX_CONCURRENCY = 128
    root = pathlib.Path(
        "/datasets/toaugspurger/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-SUBSET.zarr"
    )

    # clone the static metadata
    static = [
        ".zattrs",
        ".zmetadata",
        ".zgroup",
    ]
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    coros = [get_if_not_exists(fs, f"{PREFIX}/{x}", root / x, sem) for x in static]
    logger.info("Getting static metadata")
    await asyncio.gather(*coros)

    coros = []

    for k, v in ds.data_vars.items():
        if v.dims[0] == "time" and len(v.dims) == 3:
            xs = [f"{PREFIX}/{k}/{i}.0.0" for i in range(N)]
        elif v.dims[0] == "time" and len(v.dims) == 4:
            xs = [f"{PREFIX}/{k}/{i}.0.0.0" for i in range(N)]
        elif len(v.dims) == 2:
            # e.g. angle_of_sub_gridscale_orography
            xs = [f"{PREFIX}/{k}/0.0"]
        else:
            assert False, f"Unknown data_var: {k}"
        for x in xs:
            rpath = root / k / x.split("/")[-1]
            coros.append(get_if_not_exists(fs, x, rpath, sem))
        coros.append(
            get_if_not_exists(fs, f"{PREFIX}/{k}/.zarray", root / k / ".zarray", sem)
        )
        coros.append(
            get_if_not_exists(fs, f"{PREFIX}/{k}/.zattrs", root / k / ".zattrs", sem)
        )

    logger.info("Getting array data")
    await asyncio.gather(*coros)

    single_coordinates = [
        "latitude",
        "longitude",
        "level",
    ]

    coros = []
    for coord in single_coordinates:
        # these are single chunked
        coros.append(
            get_if_not_exists(fs, f"{PREFIX}/{coord}/0", root / coord / "0", sem)
        )
        coros.append(
            get_if_not_exists(
                fs, f"{PREFIX}/{coord}/.zarray", root / coord / ".zarray", sem
            )
        )
        coros.append(
            get_if_not_exists(
                fs, f"{PREFIX}/{coord}/.zattrs", root / coord / ".zattrs", sem
            )
        )

    # time is chunked so we need to actually list it
    time_chunks = await fs._ls(f"{PREFIX}/time")
    for time_chunk in time_chunks:
        coros.append(
            get_if_not_exists(
                fs, time_chunk, root / "time" / time_chunk.split("/")[-1], sem
            )
        )

    logger.info("Getting coordinates")
    await asyncio.gather(*coros)


if __name__ == "__main__":
    asyncio.run(clone())
