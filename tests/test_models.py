import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from llmcord_core.config import ModelProfile, Settings
from llmcord_core.models import ImageInput, ModelGateway, TurnMessage


def gateway(provider):
    profile = ModelProfile(provider, "test-model", 16000, True,
        base_url="http://localhost/v1" if provider == "compatible" else None)
    settings = Settings("token", None, ":memory:", 90, {"test": profile},
        "test", "test", "test", {"max_output_tokens": 700})
    return ModelGateway(settings)


class ModelAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_responses_text_and_schema(self):
        model = gateway("openai")
        create = AsyncMock(return_value=SimpleNamespace(output_text='{"speakers":[1]}'))
        model.clients["test"] = SimpleNamespace(responses=SimpleNamespace(create=create))
        prompt = [TurnMessage("user", "hello", [ImageInput("image/png", b"png")])]
        self.assertEqual(await model.text("dialogue", "rules", prompt), '{"speakers":[1]}')
        self.assertEqual((await model.structured("director", "rules", prompt,
            "choose_speakers", {"type": "object"}))["speakers"], [1])
        self.assertEqual(create.call_args.kwargs["input"][0]["content"][1]["type"], "input_image")

    async def test_anthropic_messages_text_and_tool(self):
        model = gateway("anthropic")
        create = AsyncMock(side_effect=[
            SimpleNamespace(content=[SimpleNamespace(type="text", text="Hello")]),
            SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="choose_speakers",
                input={"speakers": [1]})]),
        ])
        model.clients["test"] = SimpleNamespace(messages=SimpleNamespace(create=create))
        self.assertEqual(await model.text("dialogue", "rules", [TurnMessage("user", "hi")]), "Hello")
        self.assertEqual((await model.structured("director", "rules", [TurnMessage("user", "hi")],
            "choose_speakers", {"type": "object"}))["speakers"], [1])
        self.assertEqual(create.call_args.kwargs["tool_choice"]["type"], "tool")

    async def test_compatible_chat_and_provider_failure(self):
        model = gateway("compatible")
        create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"speakers":[1]}'))]))
        model.clients["test"] = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        self.assertEqual((await model.structured("director", "rules", [TurnMessage("user", "hi")],
            "choose_speakers", {"type": "object"}))["speakers"], [1])
        create.side_effect = RuntimeError("provider unavailable")
        with self.assertRaises(RuntimeError):
            await model.text("dialogue", "rules", [TurnMessage("user", "hi")])


if __name__ == "__main__":
    unittest.main()
