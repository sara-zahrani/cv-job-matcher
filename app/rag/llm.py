"""
Chat model client, through OpenRouter's OpenAI-compatible API.

Same client library and base URL as the embedder; only the endpoint and
the model differ. This is the "generate" step of RAG.
"""

from openai import OpenAI

from app.config import settings


class LLMClient:
    def __init__(self, model: str = settings.chat_model, timeout: float = 60.0) -> None:
        settings.require_api_key()
        self.model = model
        self.client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            # Fail loudly after this many seconds instead of hanging forever.
            timeout=timeout,
            max_retries=0,
        )

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Low temperature: we want a consistent assessment, not creativity.
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The model returned an empty response.")
        return content


if __name__ == "__main__":
    llm = LLMClient()
    print(f"model: {llm.model}\n")
    print(llm.generate("You answer in one short sentence.", "What is retrieval-augmented generation?"))
