# Genie Space Embed

A Dash application that embeds Databricks AI/BI Genie as a chat interface, deployed as a Databricks App.

## Authorization Architecture

This app uses two authorization modes:

### GenieMetadataClient (Service Principal)

Fetches Genie Space metadata (title, description, sample questions) on page load.

- **Auth**: Service Principal (oauth-m2m) via `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` env vars
- **Why SP?**: The Genie Space metadata API (`GET /api/2.0/genie/spaces/{id}`) requires the `genie` OAuth scope, which is not currently a valid scope for Databricks Apps OBO tokens.

**SP permissions required:**

| Resource | Permission |
|----------|-----------|
| Genie Space | `CAN_EDIT` |
| UC Catalog | `USE CATALOG` |
| UC Schema | `USE SCHEMA` |
| UC Tables (underlying the Genie Space) | `SELECT` |

### LLM Insights (Service Principal)

Calls a model serving endpoint to generate insights from query results.

- **Auth**: Service Principal (same as above)
- **Why SP?**: The model serving API requires the `model-serving` scope, which is not a valid scope for Databricks Apps OBO tokens.

**SP permissions required:**

| Resource | Permission |
|----------|-----------|
| Serving Endpoint | `CAN_QUERY` |

### GenieQueryClient (On-Behalf-Of User)

Executes Genie queries as the logged-in user via their OBO token.

- **Auth**: OBO token from `X-Forwarded-Access-Token` request header
- **App OAuth scopes**: `dashboards.genie`, `sql`

**User permissions required:**

| Resource | Permission |
|----------|-----------|
| Genie Space | `CAN_RUN` (or higher) |
| UC Tables (underlying the Genie Space) | `SELECT` |

## Deployment

1. Clone to a Databricks workspace directory:
   ```
   /Workspace/Users/<you>/genie_space_embed
   ```

2. Create a Databricks App and configure `app.yaml`:
   ```yaml
   command:
   - "python"
   - "app.py"

   env:
   - name: "SPACE_ID"
     valueFrom: "genie-space"
   - name: "SERVING_ENDPOINT_NAME"
     valueFrom: "serving-endpoint"

   authorization:
     type: oauth
     scopes:
       - genie
       - sql
   ```
   The `authorization.scopes` section is required — it controls which scopes the OAuth flow requests for OBO tokens. Without it, queries will fail with `403 Invalid scope`.

3. In the App settings, add resources:
   - **genie-space**: Your Genie Space ID (SP gets `CAN_EDIT`)
   - **serving-endpoint**: A model serving endpoint for insights generation (SP gets `CAN_QUERY`)

4. Grant the App's Service Principal access to the UC tables underlying the Genie Space:
   ```sql
   GRANT USE CATALOG ON CATALOG <catalog> TO `<sp-name>`;
   GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<sp-name>`;
   GRANT SELECT ON SCHEMA <catalog>.<schema> TO `<sp-name>`;
   ```

5. Grant end users access to the Genie Space (`CAN_RUN`) and `SELECT` on the underlying UC tables.

6. Deploy:
   ```bash
   databricks apps deploy <app-name> --source-code-path /Workspace/Users/<you>/genie_space_embed
   ```

## Resources

- [Genie Conversation APIs](https://docs.databricks.com/api/workspace/genie)
- [Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/)
