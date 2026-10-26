"""Offline CLI and actual SDK HTTP-contract checks; no cloud resources."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

import requests

import verda_cli


class VerdaCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env_file = Path(self.directory.name) / ".env"
        self.env_file.write_text("")
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def call(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            status = verda_cli.main([*args, "--env-file", str(self.env_file)])
        return status, json.loads(output.getvalue()), output.getvalue()

    def credentials(self):
        self.env_file.write_text(
            'VERDA_CLIENT_ID=file-client-id\n'
            'VERDA_CLIENT_SECRET="literal-${KEEP_THIS}-secret"\n'
            'VERDA_LOCATION=FIN-03\n'
        )

    def test_offline_doctor_never_constructs_an_authenticated_client(self):
        with patch("verda_cli.VerdaClient") as client:
            status, result, _ = self.call("doctor")
        client.assert_not_called()
        self.assertEqual(status, 0)
        self.assertTrue(result["ok"])
        self.assertFalse(result["data"]["ready_for_api"])
        self.assertEqual(result["data"]["authentication"], "not_checked")

    def test_missing_keys_fail_before_network(self):
        with patch("verda_cli.VerdaClient") as client:
            for command in ("catalog", "clusters", "instances", "availability", "images"):
                with self.subTest(command=command):
                    status, result, _ = self.call(command)
                    self.assertEqual(status, 2)
                    self.assertEqual(result["error"]["code"], "credentials_missing")
            self.assertEqual(self.call("doctor", "--online")[0], 2)
        client.assert_not_called()

    def test_env_literal_values_and_process_environment_precedence(self):
        self.credentials()
        with patch.dict(os.environ, VERDA_CLIENT_ID="process-client-id"):
            with patch("verda_cli.VerdaClient") as client:
                client.return_value.locations.get.return_value = []
                status, result, raw = self.call("doctor", "--online")
        client.assert_called_once_with(
            "process-client-id", "literal-${KEEP_THIS}-secret", base_url=verda_cli.API_URL)
        self.assertEqual(status, 0)
        self.assertTrue(result["data"]["authenticated"])
        self.assertNotIn("process-client-id", raw)
        self.assertNotIn("literal-${KEEP_THIS}-secret", raw)

    def test_real_sdk_auth_and_cluster_read_use_bounded_requests(self):
        self.credentials()
        observed = []
        cluster = {
            "id": "cluster-1", "hostname": "test-cluster", "description": "",
            "status": "running", "created_at": "2026-10-26T00:00:00Z",
            "location": "FIN-03", "cluster_type": "16B200",
            "worker_nodes": [{"id": "node-1", "hostname": "worker-1",
                              "status": "running", "private_ip": "10.0.0.1"}],
            "shared_volumes": [], "ssh_key_ids": [], "ip": "198.51.100.1",
        }

        def send(session, request, **kwargs):
            path = urlparse(request.url).path
            observed.append((request.method, path, kwargs["timeout"]))
            if path == "/v1/oauth2/token":
                self.assertEqual(request.method, "POST")
                self.assertEqual(json.loads(request.body)["client_secret"],
                                 "literal-${KEEP_THIS}-secret")
                data = {"access_token": "fake-access", "refresh_token": "fake-refresh",
                        "scope": "cloud-api-v1", "token_type": "Bearer", "expires_in": 3600}
            else:
                self.assertEqual(request.method, "GET")
                self.assertEqual(request.headers["Authorization"], "Bearer fake-access")
                self.assertEqual(path, "/v1/clusters/cluster-1")
                data = cluster
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(data).encode()
            response.request = request
            return response

        with patch.object(requests.sessions.Session, "send", send):
            status, result, raw = self.call("clusters", "cluster-1")
        self.assertEqual(status, 0)
        self.assertEqual(result["data"]["worker_nodes"][0]["hostname"], "worker-1")
        self.assertEqual(observed, [
            ("POST", "/v1/oauth2/token", (5, 30)),
            ("GET", "/v1/clusters/cluster-1", (5, 30)),
        ])
        self.assertNotIn("fake-access", raw)
        self.assertNotIn("fake-refresh", raw)

    def test_availability_and_images_use_sdk_read_methods(self):
        self.credentials()
        with patch("verda_cli.VerdaClient") as constructor:
            client = constructor.return_value
            client.clusters.is_available.return_value = False
            status, result, _ = self.call(
                "availability", "--type", "16B200", "--location", "FIN-03")
            self.assertEqual(status, 0)
            self.assertFalse(result["data"]["available"])
            client.clusters.is_available.assert_called_once_with("16B200", location_code="FIN-03")
            client.clusters.get_cluster_images.return_value = ["cluster-image"]
            status, result, _ = self.call("images", "--kind", "cluster", "--type", "16B200")
            self.assertEqual(result["data"], ["cluster-image"])
            client.clusters.get_cluster_images.assert_called_once_with("16B200")
            client.clusters.create.assert_not_called()

    def test_auth_errors_and_timeouts_do_not_emit_raw_server_responses(self):
        self.credentials()

        def reject(session, request, **kwargs):
            response = requests.Response()
            response.status_code = 401
            response._content = json.dumps({
                "code": "UNAUTHORIZED",
                "message": "literal-${KEEP_THIS}-secret fake-access fake-refresh",
            }).encode()
            return response

        with patch.object(requests.sessions.Session, "send", reject):
            status, result, raw = self.call("doctor", "--online")
        self.assertEqual(status, 1)
        self.assertEqual(result["error"]["code"], "api_error")
        self.assertNotIn("fake-access", raw)
        self.assertNotIn("literal-", raw)
        with patch.object(requests.sessions.Session, "send", side_effect=requests.Timeout):
            status, result, _ = self.call("doctor", "--online")
        self.assertEqual(status, 1)
        self.assertEqual(result["error"]["code"], "timeout")

    def test_json_output_matches_stdout(self):
        target = Path(self.directory.name) / "inventory.json"
        status, result, _ = self.call("doctor", "--output", str(target))
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(target.read_text()), result)


if __name__ == "__main__":
    unittest.main()
