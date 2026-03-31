import logging
import sys
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


def setup_logging(level: str = "INFO") -> None:
    """
    Configure structured logging for the application.

    Sets up a consistent log format with timestamps, levels,
    and logger names for easy debugging and monitoring.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request-level tracing.

    For every request, logs:
      - Request method and path
      - Response status code
      - Total request latency
      - A unique request ID for correlation
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        logger = logging.getLogger("api.request")

        start_time = time.perf_counter()

        logger.info(
            "[%s] %s %s",
            request_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "[%s] %s %s -> %d (%.0fms)",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                latency_ms,
            )

            response.headers["X-Request-ID"] = request_id
            return response

        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "[%s] %s %s -> ERROR (%.0fms)",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
                exc_info=True,
            )
            raise
