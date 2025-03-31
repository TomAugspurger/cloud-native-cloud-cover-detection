"""
Download source imagery a store we control.

Preprocess to something that can be used for training.

The persisted dataset will be something like a Zarr store.

"""

import pathlib
import io
import pandas as pd
import boto3
import obstore.auth.boto3
import obstore.store

SINK = pathlib.Path("/datasets/toaugspurger")


def build_store() -> obstore.store.S3Store:
    session = boto3.Session(profile_name="sc")
    credential_provider = obstore.auth.boto3.Boto3CredentialProvider(session)
    store = obstore.store.S3Store(
        bucket="radiantearth",
        endpoint="https://data.source.coop",
        credential_provider=credential_provider,
    )
    return store


async def list_chip_ids(store: obstore.store.S3Store) -> list[str]:
    r = await store.get_async("cloud-cover-detection-challenge/final/public/train_metadata.csv")
    df = pd.read_csv(io.BytesIO(r.bytes()))
    return df["chip_id"].tolist()


async def get_train(store: obstore.store.S3Store):
    # cloud-cover-detection-challenge/final/public/train_labels.zip
    # cloud-cover-detection-challenge/final/public/train_metadata.csv
    ...


def build_keys_for_id(chip_id: str) -> list[str]:
    files = [
        "B02.tif",
        "B03.tif",
        "B04.tif",
        "B08.tif",
    ]
    return [
        f"cloud-cover-detection-challenge/final/public/train_features/{chip_id}/{file}"
        for file in files
    ]


async def download_chips(store: obstore.store.S3Store, sink, chip_ids: list[str]) -> None:
    # process concurrently
    for chip_id in chip_ids:
        keys = build_keys_for_id(chip_id)
        for key in keys:
            r = await store.get_async(key)
            with open(f"{chip_id}/{key}", "wb") as f:
                f.write(r.bytes())


# cloud-cover-detection-challenge/final/public/train_metadata.csv
#
# prefixes = store.list_with_delimiter(
#     prefix="cloud-cover-detection-challenge/final/public/train_features"
# )
