"""Small read-only Verda CLI backed by the official Python SDK.

Run: uv run --extra verda python verda_cli.py doctor
Cloud API credentials come from this project's .env or the process environment.
"""

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

from dotenv import dotenv_values
import requests
from verda import VerdaClient
from verda.exceptions import APIException

ROOT = Path(__file__).resolve().parent
API_URL = "https://api.verda.com/v1"
COMMANDS = ("doctor", "catalog", "clusters", "instances", "availability",
            "images", "ssh-keys", "balance")


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Unsupported SDK result type: {type(value).__name__}")


def query(client, args):
    """Use SDK read methods only; job submission belongs to the cluster scheduler."""
    if args.command == "doctor":
        locations = client.locations.get()
        return {"authenticated": True, "location_count": len(locations)}
    if args.command == "catalog":
        return {
            "locations": client.locations.get(),
            "instance_types": client.instance_types.get(),
            "cluster_types": client.cluster_types.get(),
        }
    if args.command == "clusters":
        return (client.clusters.get_by_id(args.resource_id) if args.resource_id
                else client.clusters.get(status=args.status))
    if args.command == "instances":
        return (client.instances.get_by_id(args.resource_id) if args.resource_id
                else client.instances.get(status=args.status))
    if args.command == "availability":
        service = client.clusters if args.kind == "cluster" else client.instances
        if args.type:
            return {"type": args.type, "location": args.location,
                    "available": service.is_available(args.type, location_code=args.location)}
        return service.get_availabilities(location_code=args.location)
    if args.command == "images":
        return (client.clusters.get_cluster_images(args.type) if args.kind == "cluster"
                else client.images.get(instance_type=args.type))
    if args.command == "ssh-keys":
        return [{"id": key.id, "name": key.name} for key in client.ssh_keys.get()]
    if args.command == "balance":
        balance = client.balance.get()
        return {"amount": balance.amount, "currency": balance.currency}
    raise ValueError("Unknown command")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("resource_id", nargs="?", help="Optional ID for clusters/instances")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--online", action="store_true", help="Authenticate during doctor")
    parser.add_argument("--kind", choices=("cluster", "instance"), default="cluster")
    parser.add_argument("--type", help="Type slug for availability/images, from the live catalog")
    parser.add_argument("--location", help="Location code; defaults to VERDA_LOCATION or FIN-03")
    parser.add_argument("--status", help="Optional status filter for cluster/instance lists")
    parser.add_argument("--timeout", type=float, default=30, help="Read timeout in seconds")
    parser.add_argument("--output", type=Path, help="Also save the JSON response locally")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.resource_id and args.command not in ("clusters", "instances"):
        parser.error("A resource ID is only valid for clusters/instances")
    if args.online and args.command != "doctor":
        parser.error("--online is only needed for doctor")
    if args.output and args.output.resolve() == args.env_file.resolve():
        parser.error("--output must not overwrite the credential file")

    credentials = {}
    result = {"ok": False, "command": args.command, "api_url": API_URL,
              "checked_utc": datetime.now(timezone.utc).isoformat()}
    exit_code = 0
    try:
        # No shell execution or interpolation; process environment takes precedence.
        values = dotenv_values(args.env_file, interpolate=False) if args.env_file.is_file() else {}
        credentials = {key: (os.environ.get(key, values.get(key)) or "").strip()
                       for key in ("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET")}
        args.location = args.location or os.environ.get(
            "VERDA_LOCATION", values.get("VERDA_LOCATION") or "FIN-03")
        present = {key: bool(value) for key, value in credentials.items()}
        if args.command == "doctor" and not args.online:
            result.update(ok=True, data={
                "python": platform.python_version(), "sdk_version": version("verda"),
                "dotenv_version": version("python-dotenv"),
                "env_file": str(args.env_file.resolve()), "env_file_exists": args.env_file.is_file(),
                "credential_fields_present": present, "ready_for_api": all(present.values()),
                "authentication": "not_checked", "location": args.location,
                "next": "After filling .env, run doctor --online, then clusters and catalog.",
            })
        elif not all(present.values()):
            exit_code = 2
            result["error"] = {
                "code": "credentials_missing",
                "message": "Fill VERDA_CLIENT_ID and VERDA_CLIENT_SECRET in the local .env.",
                "missing": [key for key, exists in present.items() if not exists],
            }
        else:
            original_request = requests.sessions.Session.request

            def bounded_request(session, method, url, **kwargs):
                # SDK 1.25.0 has no client timeout option; cover OAuth and resource reads.
                kwargs.setdefault("timeout", (5, args.timeout))
                return original_request(session, method, url, **kwargs)

            with patch.object(requests.sessions.Session, "request", bounded_request):
                # Construct only for online commands: SDK construction authenticates eagerly.
                client = VerdaClient(credentials["VERDA_CLIENT_ID"],
                                     credentials["VERDA_CLIENT_SECRET"], base_url=API_URL)
                result.update(ok=True, data=query(client, args))
    except APIException:
        exit_code = 1
        result["error"] = {"code": "api_error",
                           "message": "Verda rejected the request; check credentials, permissions and resource IDs."}
    except requests.exceptions.Timeout:
        exit_code = 1
        result["error"] = {"code": "timeout", "message": "Verda API request timed out."}
    except requests.exceptions.RequestException:
        exit_code = 1
        result["error"] = {"code": "connection_error",
                           "message": "Could not complete the Verda API request."}
    except Exception as exc:
        exit_code = 1
        result["error"] = {"code": type(exc).__name__,
                           "message": "Client/configuration error; raw SDK responses are not printed."}

    rendered = json.dumps(result, indent=2, default=json_default)
    for value in credentials.values():
        if value:
            rendered = rendered.replace(value, "[redacted]")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
