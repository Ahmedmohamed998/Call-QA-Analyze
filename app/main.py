import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import LLMProviderType, get_settings
from app.middleware.observability import RequestTracingMiddleware, setup_logging
from app.providers.azure_openai_provider import AzureOpenAIProvider
from app.providers.base import LLMProvider
from app.providers.bedrock_claude_provider import BedrockClaudeProvider
from app.routes.analyze import router as analyze_router, set_analyzer
from app.services.analyzer import CallAnalyzer
from app.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# Module-level reference for cleanup
_provider: LLMProvider | None = None


def _create_provider(settings) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    if settings.llm_provider == LLMProviderType.AZURE_OPENAI:
        return AzureOpenAIProvider(settings)
    elif settings.llm_provider == LLMProviderType.BEDROCK_CLAUDE:
        return BedrockClaudeProvider(settings)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes the LLM provider, prompt builder, and analyzer on startup.
    Cleans up resources on shutdown.
    """
    global _provider

    # Load settings and setup logging
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("=" * 60)
    logger.info("Call QA Analyzer starting up")
    logger.info("LLM Provider: %s", settings.llm_provider.value)
    logger.info("=" * 60)

    # Initialize components
    _provider = _create_provider(settings)
    prompt_builder = PromptBuilder()
    analyzer = CallAnalyzer(provider=_provider, prompt_builder=prompt_builder)

    # Inject analyzer into routes
    set_analyzer(analyzer)

    logger.info("Application initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down...")
    if _provider:
        await _provider.close()
    logger.info("Shutdown complete")


# --------------------------------------------------------------------------
# Create FastAPI application
# --------------------------------------------------------------------------

app = FastAPI(
    title="Call QA Analyzer",
    description=(
        "AI-powered call quality analysis system for healthcare operations. "
        "Analyzes phone call transcripts to detect compliance issues, "
        "evaluate agent performance, and determine escalation needs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestTracingMiddleware)

# Routes
app.include_router(analyze_router, tags=["Analysis"])


@app.get(
    "/health",
    summary="Health check",
    description="Returns the health status of the service.",
    tags=["System"],
)
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "call-qa-analyzer",
        "version": "1.0.0",
    }
