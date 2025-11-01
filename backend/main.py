"""FastAPI backend entrypoint for the Deneme-003 project."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(title="Deneme-003 Backend", version="0.1.0")

    @app.get("/", summary="Servis durumu")
    async def root() -> dict[str, str]:
        """Return a basic status payload for sanity checks."""
        return {"status": "ok", "message": "Deneme-003 backend çalışıyor."}

    @app.get("/health", summary="Sağlık kontrolü")
    async def health() -> dict[str, str]:
        """Expose a dedicated health-check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()
