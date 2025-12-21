"""Application settings and configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server configuration
    backend_port: int = 8000

    # RediAI integration
    rediai_mode: str = "mock"  # "mock" or "real"
    rediai_base_url: str = "http://localhost:8080"
    rediai_health_path: str = "/health"

    @property
    def rediai_health_url(self) -> str:
        """Construct full RediAI health URL."""
        return f"{self.rediai_base_url}{self.rediai_health_path}"


# Global settings instance
settings = Settings()

