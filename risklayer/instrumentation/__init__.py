from risklayer.instrumentation.openai import patch_openai, unpatch_openai
from risklayer.instrumentation.langchain import patch_langchain, unpatch_langchain
from risklayer.integrations.autogen_hook import patch_autogen, unpatch_autogen

def patch_all():
    """Automatically instruments all supported LLM SDKs (OpenAI, LangChain, AutoGen)."""
    patch_openai()
    patch_langchain()
    patch_autogen()

def unpatch_all():
    """Removes all risklayer instrumentation."""
    unpatch_openai()
    unpatch_langchain()
    unpatch_autogen()
