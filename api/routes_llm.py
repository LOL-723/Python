from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm.client import llm_client
from schemas.llm import ChatRequest, ChatResponse, PersonInfoOutput


router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        answer = llm_client.chat(
            user_message=req.message,
            system_prompt=req.system_prompt,
        )
        return ChatResponse(answer=answer)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM request failed: {str(e)}"
        )


@router.post("/stream_chat")
def stream_chat(req: ChatRequest):
    try:
        if not req.message or not req.message.strip():
            raise ValueError("message cannot be empty")

        return StreamingResponse(
            llm_client.stream_chat(
                user_message=req.message,
                system_prompt=req.system_prompt,
            ),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM stream request failed: {str(e)}"
        )


@router.post("/extract_person", response_model=PersonInfoOutput)
def extract_person(req: ChatRequest):
    try:
        result = llm_client.structured_chat(
            user_message=req.message,
            output_schema=PersonInfoOutput,
            system_prompt=req.system_prompt or (
                "You are an information extraction assistant. "
                "Extract person information from the target sentence in the user message. "
                "If the user message contains a Chinese or English colon, the target sentence is after the colon. "
                "For example, from '小红今年 18 岁，会 Java' return "
                '{"name":"小红","age":18,"skills":["Java"]}. '
                "The name is usually the text before '今年'. "
                "Return JSON only."
            ),
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM structured request failed: {str(e)}"
        )
