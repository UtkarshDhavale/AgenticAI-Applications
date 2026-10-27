from dotenv import load_dotenv

from livekit import agents
from langchain_core.messages import ToolMessage
from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, inference
from livekit.plugins import langchain as lk_langchain

from config import settings
from langchain_agent import clinic_agent

_ = load_dotenv()


def _is_tool_token(item: object) -> bool:
    """True for the (ToolMessage, metadata) pairs astream emits from the tools node."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage)


class SpeechOnlyGraph:
    """Wraps the graph so tool return values never reach TTS."""

    def __init__(self, graph: object) -> None:
        self._graph = graph

    def astream(self, *args: object, **kwargs: object):
        inner = self._graph.astream(*args, **kwargs)

        async def _filtered():
            async for item in inner:
                if not _is_tool_token(item):
                    yield item

        return _filtered()


class ClinicVoiceAgent(Agent):
    """Clinic voice agent backed by a LangChain agent graph."""

    def __init__(self) -> None:
        super().__init__(
            instructions="You are the voice assistant for Sahyadri Family Clinic, a fictional demo clinic. Keep answers brief. Do not diagnose or prescribe.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(clinic_agent)),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(
            model="inworld/inworld-tts-2",
            voice="Ashley",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "mode": "fixed",
                "min_delay": 0.5,
                "max_delay": 6.0,
            },
        ),
    )

    await session.start(agent=ClinicVoiceAgent(), room=ctx.room)

    await session.generate_reply(user_input="Greet the caller as Sahyadri Family Clinic's demo voice assistant and ask how you can help.")


if __name__ == "__main__":
    agents.cli.run_app(server)
