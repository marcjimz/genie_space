"""
Genie Space API clients with dual-auth architecture.

Authorization Architecture
==========================

GenieMetadataClient (Service Principal)
  Reads Genie Space metadata: title, description, sample questions.
  - Auth: SP (oauth-m2m) via DATABRICKS_CLIENT_ID/SECRET env vars
  - SP permissions required:
      * Genie Space resource: CAN_EDIT
      * UC catalog: USE CATALOG
      * UC schema: USE SCHEMA
      * UC tables: SELECT (on the tables underlying the Genie Space)
  - Note: GET /api/2.0/genie/spaces/{id} requires the 'genie' OAuth scope,
    which is NOT a valid scope for Databricks Apps OBO tokens. Setting
    'genie' in app.yaml authorization.scopes maps to 'dashboards.genie'
    (query APIs only), not the management-level scope. SP auth is
    therefore required for this endpoint.
  - If include_serialized_space=true fails (needs UC table access), the
    client falls back to a basic fetch (title/description only).

GenieQueryClient (On-Behalf-Of User)
  Executes Genie queries as the logged-in user.
  - Auth: OBO token from X-Forwarded-Access-Token request header
  - App OAuth scopes: dashboards.genie, sql
  - User permissions required:
      * Genie Space resource: CAN_RUN (or higher)
      * UC tables: SELECT (on the tables underlying the Genie Space)
  - Falls back to SP auth when no OBO token is present (local dev)

LLM Insights (Service Principal)
  Calls model serving endpoint for data insights generation.
  - Auth: SP (oauth-m2m) — same as GenieMetadataClient
  - SP permissions required:
      * Serving endpoint resource: CAN_QUERY
  - Note: The 'model-serving' OAuth scope required by serving endpoints
    is not a valid scope for Databricks Apps OBO tokens.

TODO: When 'genie' and 'model-serving' OAuth scopes are formally added
      to Databricks Apps, migrate to OBO auth and remove the SP dependency.
"""
import json
import pandas as pd
import time
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import backoff
from databricks.sdk import WorkspaceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SPACE_ID = os.environ.get("SPACE_ID")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")


@dataclass
class GenieResponse:
    """Structured response from Genie containing all available data."""
    text_response: Optional[str] = None
    sql_query: Optional[str] = None
    sql_description: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    data_summary: Optional[str] = None
    status: str = "OK"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# GenieMetadataClient — SP auth for space metadata
# ---------------------------------------------------------------------------

class GenieMetadataClient:
    """Reads Genie Space metadata using Service Principal auth.

    Permissions profile:
      - Auth: oauth-m2m (picks up DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET)
      - Genie Space resource: CAN_EDIT
      - UC tables: USE_CATALOG + USE_SCHEMA + SELECT (for sample questions via serialized_space)

    Note: The Genie Space metadata API (GET /api/2.0/genie/spaces/{id}) requires
    the 'genie' OAuth scope, which is NOT a valid scope for Databricks Apps OBO
    tokens. Tested: both 'dashboards.genie' and 'genie' scopes fail — 'genie' is
    rejected as invalid by the Apps API. SP auth is therefore required.
    """

    def __init__(self):
        logger.info("[GenieMetadataClient] Initializing with SP auth")
        self._ws = WorkspaceClient()  # Uses SP creds from env

    def get_space_metadata(self, space_id: str) -> Dict[str, Any]:
        """Fetch space metadata (title, description, sample questions).

        Tries include_serialized_space=true first for sample questions.
        Falls back to basic fetch (title only) if that fails.

        Returns:
            dict with keys: title, description, sample_questions (list of str)
        """
        result = {"title": None, "description": None, "sample_questions": []}

        # Try full fetch (includes sample questions in serialized_space)
        try:
            resp = self._ws.api_client.do(
                "GET",
                f"/api/2.0/genie/spaces/{space_id}?include_serialized_space=true",
            )
            logger.info("[GenieMetadataClient] Fetched metadata with serialized_space")
            result["title"] = resp.get("title")
            result["description"] = resp.get("description")

            ss_raw = resp.get("serialized_space")
            if ss_raw:
                ss = json.loads(ss_raw)
                sq = ss.get("config", {}).get("sample_questions", [])
                for q in sq:
                    questions = q.get("question", [])
                    if questions:
                        result["sample_questions"].append(questions[0])
                logger.info(
                    f"[GenieMetadataClient] Found {len(result['sample_questions'])} sample questions"
                )
            return result
        except Exception as e:
            logger.warning(f"[GenieMetadataClient] Full fetch failed: {e}")

        # Fallback: basic metadata (no serialized_space, no table access needed)
        try:
            resp = self._ws.api_client.do(
                "GET",
                f"/api/2.0/genie/spaces/{space_id}",
            )
            result["title"] = resp.get("title")
            result["description"] = resp.get("description")
            logger.info(
                f"[GenieMetadataClient] Basic fetch OK — title: {result['title']}"
            )
        except Exception as e2:
            logger.error(f"[GenieMetadataClient] Basic fetch also failed: {e2}")

        return result


# ---------------------------------------------------------------------------
# GenieQueryClient — OBO auth for data queries
# ---------------------------------------------------------------------------

class GenieQueryClient:
    """Executes Genie queries using On-Behalf-Of (OBO) user auth.

    Permissions profile:
      - Auth: OBO token (X-Forwarded-Access-Token header) with auth_type=pat
      - App OAuth scopes: dashboards.genie, sql
      - User needs: access to the Genie Space + underlying UC tables
      - Fallback: SP auth when no OBO token is present (local dev)
    """

    def __init__(self, host: str, space_id: str, user_token: str = None):
        self.host = host
        self.space_id = space_id
        if user_token:
            h = f"https://{host}" if host and not host.startswith("https://") else host
            logger.info("[GenieQueryClient] Using OBO token")
            self._ws = WorkspaceClient(host=h, token=user_token, auth_type="pat")
        else:
            logger.info("[GenieQueryClient] No OBO token — using SP auth (local dev)")
            self._ws = WorkspaceClient()
        self._api_path = f"/api/2.0/genie/spaces/{space_id}"

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2, jitter=backoff.full_jitter,
        on_backoff=lambda d: logger.warning(
            f"Retrying in {d['wait']:.1f}s (attempt {d['tries']})"
        ),
    )
    def start_conversation(self, question: str) -> Dict[str, Any]:
        return self._ws.api_client.do(
            "POST", f"{self._api_path}/start-conversation", body={"content": question}
        )

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2, jitter=backoff.full_jitter,
        on_backoff=lambda d: logger.warning(
            f"Retrying in {d['wait']:.1f}s (attempt {d['tries']})"
        ),
    )
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        return self._ws.api_client.do(
            "POST",
            f"{self._api_path}/conversations/{conversation_id}/messages",
            body={"content": message},
        )

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2, jitter=backoff.full_jitter,
        on_backoff=lambda d: logger.warning(
            f"Retrying in {d['wait']:.1f}s (attempt {d['tries']})"
        ),
    )
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        return self._ws.api_client.do(
            "GET",
            f"{self._api_path}/conversations/{conversation_id}/messages/{message_id}",
        )

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2, jitter=backoff.full_jitter,
        on_backoff=lambda d: logger.warning(
            f"Retrying in {d['wait']:.1f}s (attempt {d['tries']})"
        ),
    )
    def get_query_result(
        self, conversation_id: str, message_id: str, attachment_id: str
    ) -> Dict[str, Any]:
        result = self._ws.api_client.do(
            "GET",
            f"{self._api_path}/conversations/{conversation_id}/messages/{message_id}"
            f"/attachments/{attachment_id}/query-result",
        )
        data_array = []
        if "statement_response" in result:
            if "result" in result["statement_response"]:
                data_array = result["statement_response"]["result"].get(
                    "data_array", []
                )
        return {
            "data_array": data_array,
            "schema": result.get("statement_response", {})
            .get("manifest", {})
            .get("schema", {}),
        }

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=5, factor=2, jitter=backoff.full_jitter,
        on_backoff=lambda d: logger.warning(
            f"Retrying in {d['wait']:.1f}s (attempt {d['tries']})"
        ),
    )
    def execute_query(
        self, conversation_id: str, message_id: str, attachment_id: str
    ) -> Dict[str, Any]:
        return self._ws.api_client.do(
            "POST",
            f"{self._api_path}/conversations/{conversation_id}/messages/{message_id}"
            f"/attachments/{attachment_id}/execute-query",
        )

    def wait_for_message_completion(
        self,
        conversation_id: str,
        message_id: str,
        timeout: int = 300,
        poll_interval: int = 2,
    ) -> Dict[str, Any]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            message = self.get_message(conversation_id, message_id)
            status = message.get("status")
            if status in ["COMPLETED", "ERROR", "FAILED"]:
                return message
            time.sleep(poll_interval)
        raise TimeoutError(f"Message processing timed out after {timeout} seconds")


# ---------------------------------------------------------------------------
# Helper: OBO WorkspaceClient for LLM calls
# ---------------------------------------------------------------------------

def make_obo_client(host: str, user_token: str = None) -> WorkspaceClient:
    """Create an OBO WorkspaceClient for non-Genie API calls (e.g. LLM serving)."""
    if user_token:
        h = f"https://{host}" if host and not host.startswith("https://") else host
        return WorkspaceClient(host=h, token=user_token, auth_type="pat")
    return WorkspaceClient()


# ---------------------------------------------------------------------------
# Public API — query functions
# ---------------------------------------------------------------------------

def _generate_data_summary(df: pd.DataFrame) -> str:
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}"]
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols[:5]:
        lines.append(
            f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}"
        )
    if len(numeric_cols) > 5:
        lines.append(f"  ... and {len(numeric_cols) - 5} more numeric columns")
    return "\n".join(lines)


def process_genie_response(
    client: GenieQueryClient, conversation_id: str, message_id: str, complete_message: dict
) -> GenieResponse:
    response = GenieResponse()

    if "content" in complete_message:
        response.text_response = complete_message.get("content", "")

    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")

        if "text" in attachment and "content" in attachment["text"]:
            response.text_response = attachment["text"]["content"]

        if "query" in attachment:
            query_info = attachment.get("query", {})
            response.sql_query = query_info.get("query", "")
            response.sql_description = query_info.get("description", None)

            try:
                query_result = client.get_query_result(
                    conversation_id, message_id, attachment_id
                )
                data_array = query_result.get("data_array", [])
                schema = query_result.get("schema", {})
                columns = [col.get("name") for col in schema.get("columns", [])]

                if data_array:
                    if not columns and len(data_array) > 0:
                        columns = [f"column_{i}" for i in range(len(data_array[0]))]
                    df = pd.DataFrame(data_array, columns=columns)
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except (ValueError, TypeError):
                            pass
                    response.data = df
                    response.data_summary = _generate_data_summary(df)
            except Exception as e:
                logger.warning(f"Failed to get query result: {e}")

    if response.text_response is None and response.data is None:
        response.text_response = "No response available"

    return response


def genie_query(
    question: str, user_token: str = None, space_id: str = None
) -> GenieResponse:
    """Main entry point for querying Genie (uses OBO auth)."""
    try:
        client = GenieQueryClient(
            host=DATABRICKS_HOST,
            space_id=space_id or SPACE_ID,
            user_token=user_token,
        )
        response = client.start_conversation(question)
        conversation_id = response.get("conversation_id")
        message_id = response.get("message_id")
        complete_message = client.wait_for_message_completion(
            conversation_id, message_id
        )
        return process_genie_response(client, conversation_id, message_id, complete_message)
    except Exception as e:
        logger.error(f"Error in genie_query: {e}")
        return GenieResponse(
            text_response=f"Sorry, an error occurred: {str(e)}. Please try again.",
            status="ERROR",
            error=str(e),
        )
