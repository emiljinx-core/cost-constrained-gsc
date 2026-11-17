import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from PROSE.utils.ollama_provider import (
    OllamaConfig,
    OllamaHTTPError,
    OllamaProvider,
    OllamaProviderError,
    OllamaResponseError,
)


class FakeHTTPResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


class OllamaProviderTest(unittest.TestCase):
    messages = [
        {"role": "system", "content": "Reply briefly."},
        {"role": "user", "content": "Say hello."},
    ]

    def test_successful_response_and_logging(self):
        completion = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello."},
            "done": True,
            "total_duration": 123,
            "unexpected_content": "must not be logged",
        }
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            response, raw, log = OllamaProvider(
                endpoint="http://example.test:11434",
                model="qwen3:8b",
                timeout=7,
                log_prompts=True,
                log_responses=True,
            ).call(self.messages, temperature=0, max_tokens=8)

        self.assertEqual(response, "Hello.")
        self.assertEqual(raw, completion)
        self.assertEqual(log["provider"], "ollama")
        self.assertEqual(log["endpoint"], "http://example.test:11434")
        self.assertEqual(log["prompt"], "Say hello.")
        self.assertEqual(log["system_prompt"], "Reply briefly.")
        self.assertEqual(log["response"], "Hello.")
        self.assertEqual(log["metadata"]["total_duration"], 123)
        self.assertNotIn("unexpected_content", log["metadata"])
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data)["options"], {
            "temperature": 0,
            "num_predict": 8,
        })

    def test_thinking_can_be_disabled(self):
        completion = {"message": {"content": "OK"}, "done": True}
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            response, _, _ = OllamaProvider(
                endpoint="http://example.test",
                think=False,
            ).call(self.messages)

        self.assertEqual(response, "OK")
        request_payload = json.loads(mocked_urlopen.call_args.args[0].data)
        self.assertFalse(request_payload["think"])

    def test_thinking_is_omitted_when_unconfigured(self):
        completion = {"message": {"content": "OK"}, "done": True}
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            OllamaProvider(endpoint="http://example.test").call(self.messages)

        request_payload = json.loads(mocked_urlopen.call_args.args[0].data)
        self.assertNotIn("think", request_payload)

    def test_openai_json_response_format_is_translated_for_ollama(self):
        completion = {"message": {"content": '{"summary": "OK"}'}, "done": True}
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            OllamaProvider(endpoint="http://example.test").call(
                self.messages,
                response_format={"type": "json_object"},
            )

        request_payload = json.loads(mocked_urlopen.call_args.args[0].data)
        self.assertEqual(request_payload["format"], "json")
        self.assertNotIn("response_format", request_payload)

    def test_prompt_logging_is_opt_in(self):
        completion = {"message": {"content": "OK"}, "done": True}
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ):
            _, _, log = OllamaProvider(endpoint="http://example.test").call(
                self.messages
            )

        self.assertIsNone(log["prompt"])
        self.assertIsNone(log["system_prompt"])
        self.assertIsNone(log["messages"])
        self.assertIsNone(log["response"])

    def test_sensitive_prompt_and_response_are_absent_by_default(self):
        sensitive = "UNIQUE_PARTICIPANT_SECRET_7f3c"
        messages = [
            {"role": "user", "content": f"Opinion: {sensitive}"},
        ]
        completion = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": f"Generated response for {sensitive}",
            },
            "done": True,
            "total_duration": 123,
        }
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ):
            _, _, log = OllamaProvider(endpoint="http://example.test").call(messages)

        serialized_log = repr(log)
        self.assertNotIn(sensitive, serialized_log)
        self.assertIsNone(log["response"])
        self.assertEqual(log["metadata"]["total_duration"], 123)
        self.assertEqual(log["params"], {})

    def test_unexpected_parameter_is_sent_but_not_logged(self):
        sensitive = "UNEXPECTED_SECRET_PARAMETER_91ab"
        completion = {
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
        }
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            _, _, log = OllamaProvider(endpoint="http://example.test").call(
                self.messages,
                sensitive_option=sensitive,
            )

        request_payload = json.loads(mocked_urlopen.call_args.args[0].data)
        self.assertEqual(request_payload["sensitive_option"], sensitive)
        self.assertNotIn(sensitive, repr(log))
        self.assertNotIn("sensitive_option", log["params"])

    def test_invalid_approved_parameter_value_is_sent_but_not_logged(self):
        sensitive = "SENSITIVE_TEMPERATURE_VALUE"
        completion = {
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
        }
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            _, _, log = OllamaProvider(endpoint="http://example.test").call(
                self.messages,
                temperature=sensitive,
            )

        request_payload = json.loads(mocked_urlopen.call_args.args[0].data)
        self.assertEqual(request_payload["options"]["temperature"], sensitive)
        self.assertNotIn(sensitive, repr(log))
        self.assertNotIn("temperature", log["params"])

    def test_endpoint_credentials_are_redacted_from_log(self):
        sensitive = "ENDPOINT_SECRET"
        completion = {
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
        }
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            _, _, log = OllamaProvider(
                endpoint=(
                    f"http://user:{sensitive}@example.test:11434/chat"
                    f"?token={sensitive}"
                )
            ).call(self.messages)

        self.assertNotIn(sensitive, repr(log))
        self.assertEqual(log["endpoint"], "http://example.test:11434")
        self.assertIn(sensitive, mocked_urlopen.call_args.args[0].full_url)

    def test_endpoint_path_is_not_logged(self):
        sensitive = "PATH_SECRET_4b21"
        completion = {
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
        }
        endpoint = f"http://example.test:11434/{sensitive}/api"
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            _, _, log = OllamaProvider(endpoint=endpoint).call(self.messages)

        self.assertEqual(log["endpoint"], "http://example.test:11434")
        self.assertNotIn(sensitive, repr(log))
        self.assertEqual(
            mocked_urlopen.call_args.args[0].full_url,
            f"{endpoint}/api/chat",
        )

    def test_prompt_and_response_logging_can_be_enabled(self):
        sensitive = "EXPLICIT_LOGGING_MARKER"
        messages = [{"role": "user", "content": sensitive}]
        completion = {
            "message": {"role": "assistant", "content": f"response-{sensitive}"},
            "done": True,
        }
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ):
            _, _, log = OllamaProvider(
                endpoint="http://example.test",
                log_prompts=True,
                log_responses=True,
            ).call(messages)

        self.assertIn(sensitive, log["prompt"])
        self.assertIn(sensitive, log["response"])

    def test_invalid_response(self):
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(b'{"message": {}}'),
        ):
            with self.assertRaises(OllamaResponseError):
                OllamaProvider(endpoint="http://example.test").call(self.messages)

    def test_empty_response(self):
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(
                b'{"message": {"content": "   "}}'
            ),
        ):
            with self.assertRaises(OllamaResponseError):
                OllamaProvider(endpoint="http://example.test").call(self.messages)

    def test_http_error(self):
        error = HTTPError(
            "http://example.test/api/chat", 500, "server error", {}, None
        )
        error.read = lambda: b'{"error":"failed"}'
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            side_effect=error,
        ):
            with self.assertRaises(OllamaHTTPError):
                OllamaProvider(endpoint="http://example.test").call(self.messages)

    def test_timeout_or_transport_error(self):
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            side_effect=URLError("timed out"),
        ):
            with self.assertRaises(OllamaProviderError):
                OllamaProvider(endpoint="http://example.test").call(self.messages)

    def test_timeout_is_passed_to_the_http_request(self):
        completion = {"message": {"content": "OK"}, "done": True}
        with patch(
            "PROSE.utils.ollama_provider.urlopen",
            return_value=FakeHTTPResponse(json.dumps(completion).encode()),
        ) as mocked_urlopen:
            OllamaProvider(endpoint="http://example.test", timeout=13).call(
                self.messages
            )
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 13)

    def test_environment_configuration(self):
        with patch.dict(
            os.environ,
            {
                "OLLAMA_BASE_URL": "http://172.21.112.1:11434/",
                "OLLAMA_MODEL": "qwen3:8b",
                "OLLAMA_TIMEOUT": "9",
            },
            clear=False,
        ):
            config = OllamaConfig.from_environment()

        self.assertEqual(config.endpoint, "http://172.21.112.1:11434")
        self.assertEqual(config.model, "qwen3:8b")
        self.assertEqual(config.timeout, 9)


if __name__ == "__main__":
    unittest.main()
