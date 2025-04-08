import pathlib
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CNCCD_")
    sink: pathlib.Path = pathlib.Path(
        "/datasets/toaugspurger/sentinel-2-cloud-cover-detection"
    )

    @pydantic.field_validator("sink", mode="after")
    def validate_sink(cls, v: pathlib.Path) -> pathlib.Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


config = Config()
