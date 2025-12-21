"""Application settings and configuration."""

from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings are validated on initialization. Invalid configuration will
    raise ValidationError with detailed error messages.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server configuration
    backend_port: int = Field(default=8000, ge=1, le=65535)

    # Database configuration
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=10, ge=0, le=50)
    db_pool_timeout: int = Field(default=30, ge=1, le=300)

    # Trace configuration
    trace_max_bytes: int = Field(default=1048576, ge=1024, le=10485760)  # 1MB default, max 10MB

    # RediAI integration
    rediai_mode: Literal["mock", "real"] = "mock"
    rediai_base_url: str = "http://localhost:8080"
    rediai_health_path: str = "/health"
    rediai_health_cache_ttl_seconds: int = Field(default=30, ge=0, le=300)

    @field_validator("rediai_base_url")
    @classmethod
    def validate_rediai_base_url(cls, v: str) -> str:
        """Validate that rediai_base_url is a valid HTTP/HTTPS URL."""
        # Use pydantic's HttpUrl for validation
        try:
            HttpUrl(v)
        except Exception as e:
            raise ValueError(f"rediai_base_url must be a valid HTTP/HTTPS URL: {e}")
        return v

    @field_validator("rediai_health_path")
    @classmethod
    def validate_rediai_health_path(cls, v: str) -> str:
        """Validate that rediai_health_path starts with /."""
        if not v.startswith("/"):
            raise ValueError("rediai_health_path must start with /")
        return v

    @property
    def rediai_health_url(self) -> str:
        """Construct full RediAI health URL."""
        return f"{self.rediai_base_url}{self.rediai_health_path}"


# Global settings instance
settings = Settings()
