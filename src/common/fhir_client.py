"""
FHIR HTTP client with pagination support.

Provides a simple interface for querying FHIR resources from the server.
"""

import os
from typing import Any

import requests

FHIR_SERVER_URL = os.environ.get("FHIR_SERVER_URL", "http://localhost:8080/fhir")

JsonObject = list[dict[str, Any]]


class FHIRClient:
    """HTTP client for FHIR server queries."""

    def __init__(self, base_url: str):
        self.fhir_store_url = base_url.rstrip('/')
        self.session = requests.Session()

    @staticmethod
    def _remove_fields(resource: dict, fields: list[str]) -> dict:
        """Remove specified fields from resource (e.g., text, meta)."""
        for field in fields:
            if field in resource:
                del resource[field]
        return resource

    def _fetch_resources_with_pagination(self, initial_resource_path: str) -> list[dict]:
        """Fetch all resources, following pagination links."""
        all_resources = []
        resource_path = initial_resource_path

        while True:
            response = self.session.get(resource_path)
            response.raise_for_status()
            resources = response.json()

            if resources.get("entry", []):
                all_resources.extend([
                    self._remove_fields(e["resource"], ["text", "meta"])
                    for e in resources["entry"]
                ])

            next_url = None
            for link in resources.get("link", []):
                if link.get("relation") == "next":
                    next_url = link.get("url")
                    break
            if not next_url:
                break
            resource_path = next_url

        return all_resources

    def search_with_pagination(self, query_string: str) -> list[dict]:
        """Execute FHIR search query and return all matching resources."""
        resource_path = f"{self.fhir_store_url}/{query_string}"
        return self._fetch_resources_with_pagination(resource_path)


def get_fhir_client() -> FHIRClient:
    """Get a FHIRClient instance using FHIR_SERVER_URL environment variable."""
    return FHIRClient(FHIR_SERVER_URL)