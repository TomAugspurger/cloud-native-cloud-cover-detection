"""
Download source imagery a store we control.

Preprocess to something that can be used for training.

The persisted dataset will be something like a Zarr store.

Usage

python -m src.cloud_cover_detection.preprocess
"""

import asyncio
import time
import structlog
import pathlib
import io
import pandas as pd
import boto3
import obstore.auth.boto3
import obstore.store

from .config import config

structlog.configure()
logger = structlog.get_logger()


def build_store(profile_name: str | None = None) -> obstore.store.S3Store:
    session = boto3.Session(profile_name=profile_name)
    credential_provider = obstore.auth.boto3.Boto3CredentialProvider(session)
    store = obstore.store.S3Store(
        bucket="radiantearth",
        endpoint="https://data.source.coop",
        credential_provider=credential_provider,
    )
    return store


# async def list_chip_ids(store: obstore.store.S3Store) -> list[str]:
#     r = await store.get_async("cloud-cover-detection-challenge/final/public/train_metadata.csv")
#     df = pd.read_csv(io.BytesIO(r.bytes()))
#     return df["chip_id"].tolist()


async def get_train_metadata(
    store: obstore.store.S3Store, sink_root: pathlib.Path
) -> pd.DataFrame:
    dst = sink_root / "train_metadata.parquet"
    if not dst.exists():
        r = await store.get_async(
            "cloud-cover-detection-challenge/final/public/train_metadata.csv"
        )
        data = await r.bytes_async()
        df = pd.read_csv(io.BytesIO(data))
        df.to_parquet(dst)
        return df
    return pd.read_parquet(dst)


async def get_crosswalk(
    store: obstore.store.S3Store, sink_root: pathlib.Path
) -> pd.DataFrame:
    dst = sink_root / "chip_id_crosswalk.parquet"
    if not dst.exists():
        r = await store.get_async(
            "cloud-cover-detection-challenge/final/chip_id_crosswalk.csv"
        )
        data = await r.bytes_async()
        df = pd.read_csv(io.BytesIO(data))
        df.to_parquet(dst)
        return df
    return pd.read_parquet(dst)


def build_keys_for_id(chip_id: str) -> list[str]:
    files = [
        "B02.tif",
        "B03.tif",
        "B04.tif",
        "B08.tif",
    ]
    keys = [
        f"cloud-cover-detection-challenge/final/public/train_features/{chip_id}/{file}"
        for file in files
    ]
    return keys


async def download_chip(
    store: obstore.store.S3Store, sink_root: pathlib.Path, chip_id: str
) -> None:
    keys = build_keys_for_id(chip_id)
    chip_root = pathlib.Path(sink_root) / chip_id
    chip_root.mkdir(exist_ok=True, parents=True)

    for key in keys:
        filename = key.split("/")[-1]
        path = pathlib.Path(sink_root) / chip_id / filename
        if not path.exists():
            logger.info("downloading", key=key)
            start = time.monotonic()
            r = await store.get_async(key)
            data = await r.bytes_async()
            duration = time.monotonic() - start
            path.write_bytes(data)
            logger.info("downloaded", key=key, duration=duration)


async def download_labels(
    store: obstore.store.S3Store, sink_root: pathlib.Path, chip_ids: str
) -> None:
    for chip_id in chip_ids:
        await download_chip_labels(store, sink_root, chip_id)


async def download_chip_labels(
    store: obstore.store.S3Store, sink_root: pathlib.Path, chip_id: str
) -> None:
    key = f"cloud-cover-detection-challenge/final/public/train_labels/{chip_id}.tif"
    path = pathlib.Path(sink_root) / chip_id / "labels.tif"
    if not path.exists():
        logger.info("downloading", key=key)
        r = await store.get_async(key)
        data = await r.bytes_async()
        path.write_bytes(data)


async def download_chips(
    store: obstore.store.S3Store, sink_root: pathlib.Path, chip_ids: list[str]
) -> None:
    # process concurrently
    for chip_id in chip_ids:
        await download_chip(store, sink_root, chip_id)


async def amain():
    store = build_store()
    metadata = await get_train_metadata(store, config.sink)
    chip_ids = metadata["chip_id"].tolist()
    n = 10

    chip_ids = chip_ids[:n]
    await download_chips(store, config.sink, chip_ids)
    await download_labels(store, config.sink, chip_ids)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
