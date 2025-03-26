import os
import json
from typing import Any, Optional
import httpx
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

# MCP server instance
mcp = FastMCP("HMNAO Celestial Engine")

# Base URL and optional subscription key environment variables
CELESTIAL_BASE_URL = os.getenv(
    "CELESTIAL_BASE_URL",
    "https://nonlive-developer-gateway.admiralty.co.uk/celestial-engine"
)
SUBSCRIPTION_KEY = os.getenv("CELESTIAL_SUBSCRIPTION_KEY", "e682c2c930b843249cf3dcf5f20f4a1f")  # or "Ocp-Apim-Subscription-Key" from OpenAPI

#
# Utility: make a request to the Celestial Engine
#
async def celestial_request(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Make a GET request to the Celestial Engine, returning JSON.
    Raises an exception if there's any HTTP or parsing error.
    """
    headers = {}
    # If your gateway requires a subscription key in a header:
    if SUBSCRIPTION_KEY:
        headers["Ocp-Apim-Subscription-Key"] = SUBSCRIPTION_KEY

    url = f"{CELESTIAL_BASE_URL}{endpoint}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

#
# 1) List Celestial Bodies
#
@mcp.tool()
async def list_celestial_bodies() -> str:
    """
    GET /celestial-bodies
    Returns a list of available celestial bodies.
    """
    data = await celestial_request("/celestial-bodies")
    return json.dumps(data, indent=2)

#
# 2) List Phenomena for a Body
#
@mcp.tool()
async def list_phenomena(body: str) -> str:
    """
    GET /celestial-bodies/{body}/phenomena
    Returns a list of available phenomena for the specified celestial body.
    """
    endpoint = f"/celestial-bodies/{body}/phenomena"
    data = await celestial_request(endpoint)
    return json.dumps(data, indent=2)

#
# 3) Retrieve Phenomena Data
#
class PhenomenaArgs(BaseModel):
    """
    The path & query params for GET /celestial-bodies/{body}/phenomena/{phenomena}
    """
    body: str = Field(..., description="Planetary body name, e.g. 'Sun' or 'Moon'")
    phenomena: str = Field(..., description="Phenomena name, e.g. 'rise-and-set', 'meridian-transit', etc.")
    latitude: str = Field(..., description="Observer's latitude in decimal degrees (e.g. '51', '-12.3')")
    longitude: str = Field(..., description="Observer's longitude in decimal degrees (e.g. '0', '-3.1')")
    startDate: str = Field(..., description="YYYY-MM-dd start date (e.g. '2024-01-01')")
    endDate: str = Field(..., description="YYYY-MM-dd end date (e.g. '2024-01-05')")
    timezone: Optional[str] = Field(None, description="Optional timezone offset in hours, e.g. '2', '-4'")
    useBst: Optional[bool] = Field(None, description="For UK locations, use BST times where applicable")
    depression: Optional[int] = Field(None, description="Depression value in decimal degrees")
    altitude: Optional[int] = Field(None, description="Altitude value in decimal degrees")

@mcp.tool()
async def get_phenomena(args: PhenomenaArgs) -> str:
    """
    GET /celestial-bodies/{body}/phenomena/{phenomena}?latitude=...&longitude=...&startDate=...&endDate=...
    Retrieves phenomena data for a given body within the specified date range.
    """
    endpoint = f"/celestial-bodies/{args.body}/phenomena/{args.phenomena}"
    params = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "startDate": args.startDate,
        "endDate": args.endDate
    }
    if args.timezone is not None:
        params["timezone"] = args.timezone
    if args.useBst is not None:
        params["useBst"] = str(args.useBst).lower()  # "true"/"false"
    if args.depression is not None:
        params["depression"] = args.depression
    if args.altitude is not None:
        params["altitude"] = args.altitude

    data = await celestial_request(endpoint, params)
    return json.dumps(data, indent=2)

#
# 4) Moon Visibility
#
class MoonVisibilityArgs(BaseModel):
    """
    The query params for GET /moon-visibility
    """
    latitude: str = Field(..., description="Observer's latitude in decimal degrees")
    longitude: str = Field(..., description="Observer's longitude in decimal degrees")
    startDate: str = Field(..., description="YYYY-MM-dd start date")
    endDate: str = Field(..., description="YYYY-MM-dd end date")
    timezone: str = Field(..., description="Timezone offset in hours, e.g. '2', '-4'")

@mcp.tool()
async def moon_visibility(args: MoonVisibilityArgs) -> str:
    """
    GET /moon-visibility?latitude=...&longitude=...&startDate=...&endDate=...&timezone=...
    Returns crescent moon visibility events for a location and date range.
    """
    endpoint = "/moon-visibility"
    params = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "startDate": args.startDate,
        "endDate": args.endDate,
        "timezone": args.timezone
    }
    data = await celestial_request(endpoint, params)
    return json.dumps(data, indent=2)

#
# Main entry point if you want to run this file directly
#
if __name__ == "__main__":
    mcp.run()
