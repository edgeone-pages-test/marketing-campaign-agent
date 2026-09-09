"""POST /stream — 营销活动策划 Agent 主入口 (SSE streaming)

架构：
- 主流程（discovery → planning → integration → content → finalize）
  由 MarketingCampaignFlow 管理，使用 @human_feedback 暂停/恢复
- 分支操作（redo_brand / redo_channel / rollback 等）
  由 handler 层拦截，直接调用 Crew，不进 Flow
"""

import asyncio

from crewai.flow.async_feedback.types import HumanFeedbackPending
from crewai.types.streaming import FlowStreamingOutput, StreamChunkType
from crewai.utilities.streaming import (
    create_async_chunk_generator,
    create_streaming_state,
    register_cleanup,
    signal_end,
    signal_error,
)

from agents._lib.flow import MarketingCampaignFlow, CampaignState, _crew_text
from agents._lib.llm import init_llm
from agents._lib.logger import make_logger
from agents._lib.persistence import (
    get_persistence, has_pending, load_pending_from_store, sync_pending_to_store,
)

# Crew imports for branch actions
from agents._crews.brand_creative_crew.brand_creative_crew import BrandCreativeCrew
from agents._crews.channel_planning_crew.channel_planning_crew import ChannelPlanningCrew
from agents._crews.integration_crew.integration_crew import IntegrationCrew
from agents._crews.content_crew.content_crew import ContentCrew

log = make_logger("Handler")


# ─── Streaming resume wrapper ────────────────────────────────────────

async def _stream_resume(flow, feedback: str) -> FlowStreamingOutput:
    """Wrap resume_async in streaming infrastructure.

    CrewAI's resume_async() doesn't return a streaming iterator, but the
    underlying Crew still emits LLMStreamChunkEvent to the event bus.
    This helper subscribes to those events — same pattern as kickoff_async.
    """
    result_holder: list = []
    task_info = {"index": 0, "name": "", "id": "", "agent_role": "", "agent_id": ""}
    state = create_streaming_state(task_info, result_holder, use_async=True)
    output_holder: list = []

    async def run():
        try:
            result = await flow.resume_async(feedback)
            result_holder.append(result)
        except Exception as e:
            if isinstance(e, HumanFeedbackPending):
                result_holder.append(e)
            else:
                signal_error(state, e, is_async=True)
        finally:
            signal_end(state, is_async=True)

    streaming = FlowStreamingOutput(
        async_iterator=create_async_chunk_generator(state, run, output_holder)
    )
    register_cleanup(streaming, state)
    output_holder.append(streaming)
    return streaming


# ─── Streaming loop (reusable async generator) ──────────────────────

def _chunk_fields(chunk) -> tuple[str, str]:
    """Extract (agent_role, text) from a streaming item.

    CrewAI >= 1.15 yields channel-based ``StreamFrame`` objects from
    ``Flow.kickoff_async()`` (via ``astream()``), while resume streams built
    with ``create_async_chunk_generator`` still yield legacy ``StreamChunk``
    objects. Support both shapes.
    """
    data = getattr(chunk, "data", None)
    if isinstance(data, dict):
        # New public frame API: StreamFrame.
        # Only LLM text chunks carry content we should show (skip thinking
        # chunks and tool-call frames).
        if getattr(chunk, "type", "") != "llm_stream_chunk":
            return "", ""
        if data.get("tool_call"):
            return "", ""
        return (data.get("agent_role") or "").strip(), data.get("chunk") or ""
    # Legacy StreamChunk
    if getattr(chunk, "chunk_type", None) == StreamChunkType.TEXT:
        return (getattr(chunk, "agent_role", "") or "").strip(), chunk.content or ""
    return "", ""


async def _consume_stream(streaming, context, phase_before: str):
    """Consume a FlowStreamingOutput and yield SSE events.

    Handles agent detection, phase transitions, parallel lanes, and card clearing.
    Yields: SSE event dicts (already wrapped via context.utils.sse).
    After exhausting, caller should read `result` attributes from the returned state.

    Note: Generated content is NOT saved as messages — it's persisted via Flow state
    and exposed to the frontend through card_update events. Only user messages are stored.
    """
    prev_agent = ""
    current_content = ""
    agent_contents: list[tuple[str, str]] = []
    in_parallel = False
    planning_emitted = False

    async for chunk in streaming:
        raw_agent, text = _chunk_fields(chunk)
        agent_role = _normalize_agent(raw_agent)

        # Detect agent switch
        if agent_role and agent_role != prev_agent:
            if prev_agent and current_content:
                agent_contents.append((prev_agent, current_content))
                current_content = ""
            if prev_agent:
                lane = _agent_to_lane(prev_agent)
                yield context.utils.sse({"type": "agent_end", "agent": prev_agent, **({"lane": lane} if lane else {})})

            # Detect phase transitions based on agent role
            lane = _agent_to_lane(agent_role)

            if lane in ("brand", "channel") and not planning_emitted:
                planning_emitted = True
                yield context.utils.sse({"type": "phase_change", "phase": "planning", "progress": _phase_progress("planning")})
                yield context.utils.sse({"type": "parallel_start", "lanes": ["brand", "channel"]})
                in_parallel = True

            if lane == "channel" and _agent_to_lane(prev_agent) == "brand":
                yield context.utils.sse({"type": "card_update", "card": "brand_creative", "data": {"raw": agent_contents[-1][1] if agent_contents else ""}})

            if not lane and in_parallel:
                in_parallel = False
                yield context.utils.sse({"type": "parallel_end"})

            if agent_role == "chief_strategist" and phase_before not in ("discovery", "finalize"):
                yield context.utils.sse({"type": "phase_change", "phase": "integration", "progress": _phase_progress("integration")})
                yield context.utils.sse({"type": "card_update", "card": "strategy", "data": {"raw": "", "content": ""}})
            elif agent_role == "copywriter":
                yield context.utils.sse({"type": "phase_change", "phase": "content", "progress": _phase_progress("content")})
                yield context.utils.sse({"type": "card_update", "card": "copywriting", "data": {"raw": ""}})

            yield context.utils.sse({"type": "agent_start", "agent": agent_role, **({"lane": lane} if lane else {})})
            prev_agent = agent_role

        if text:
            current_content += text
            yield context.utils.sse({"type": "chunk", "agent": agent_role or prev_agent, "content": text})

    # Final agent content
    if prev_agent and current_content:
        agent_contents.append((prev_agent, current_content))
    if prev_agent:
        lane = _agent_to_lane(prev_agent)
        yield context.utils.sse({"type": "agent_end", "agent": prev_agent, **({"lane": lane} if lane else {})})
    if in_parallel:
        yield context.utils.sse({"type": "parallel_end"})

    # Yield a sentinel with accumulated data (caller can detect by type)
    yield {"__stream_result__": True, "agent_contents": agent_contents, "in_parallel": in_parallel}


# ─── Branch action detection ─────────────────────────────────────────

BRANCH_ACTIONS = {
    "rollback_to_planning", "rollback_to_integration", "rollback_to_content",
}


def _parse_action(message: str) -> tuple[str, str]:
    """Parse ACTION:xxx|feedback=yyy from message. Returns (action, feedback)."""
    if not message.startswith("ACTION:"):
        return "", message
    parts = message[7:].split("|", 1)
    action = parts[0].strip()
    feedback = ""
    if len(parts) > 1 and "=" in parts[1]:
        feedback = parts[1].split("=", 1)[1].strip()
    return action, feedback


def _is_branch_action(message: str) -> bool:
    """Check if this message is a branch action that should bypass Flow."""
    if message.startswith("ACTION:"):
        action, _ = _parse_action(message)
        return action in BRANCH_ACTIONS
    return False


# ─── Main handler ────────────────────────────────────────────────────

async def handler(context):
    """POST /stream — conversation turn (streaming)."""
    conversation_id = getattr(context, "conversation_id", None)
    body = context.request.body or {}

    # Parse user input — support both text message and structured phase_action
    user_message = (
        body.get("message")
        or body.get("user_message")
        or body.get("campaign_brief")
        or ""
    ).strip()
    campaign_name = body.get("campaign_name", "")
    locale = body.get("locale", "zh")
    action = body.get("action", "send")
    phase_action = body.get("phase_action")  # e.g., {"type": "confirm"}
    card_action = body.get("card_action")    # e.g., {"target": "brand", "type": "redo", "feedback": "..."}
    iteration_feedback = body.get("iteration_feedback", "")  # e.g., "预算改为200万"

    # iteration_feedback → treat as revise_document action
    if iteration_feedback:
        user_message = f"ACTION:revise_document|feedback={iteration_feedback}"

    # Convert phase_action to feedback string
    if phase_action and isinstance(phase_action, dict):
        pa_type = phase_action.get("type", "")
        pa_feedback = phase_action.get("feedback", "")
        if pa_type == "confirm":
            user_message = user_message or "ACTION:confirm"
        elif pa_type == "keep_old":
            # Restore old content to backend state so it's used for downstream phases
            async def _keep_old_gen2():
                if pa_feedback and conversation_id:
                    await _restore_old_content(conversation_id, pa_feedback, "", context.store)
                yield context.utils.sse({"type": "done", "status": "completed"})
            return context.utils.stream_sse(_keep_old_gen2())
        elif pa_type == "rollback":
            # Generic "rollback" from frontend → map to phase-specific action
            # Determine target phase from current pending state
            rollback_target = _resolve_rollback_target(conversation_id)
            if rollback_target:
                user_message = f"ACTION:{rollback_target}"
            else:
                # Pending not in memory yet — mark as deferred rollback
                # Will be resolved after load_pending_from_store
                user_message = "ACTION:__rollback_deferred__"
        elif pa_type in BRANCH_ACTIONS:
            user_message = f"ACTION:{pa_type}" + (f"|feedback={pa_feedback}" if pa_feedback else "")
        elif pa_type and not user_message:
            user_message = f"ACTION:{pa_type}"

    # Convert card_action to feedback string
    if card_action and isinstance(card_action, dict):
        ca_target = card_action.get("target", "")  # "brand" | "channel"
        ca_type = card_action.get("type", "")      # "confirm" | "redo" | "keep_old"
        ca_feedback = card_action.get("feedback", "")
        # For keep_old, the old content is in previous_data.raw
        ca_previous_raw = ""
        if ca_type == "keep_old":
            previous_data = card_action.get("previous_data", {})
            if isinstance(previous_data, dict):
                ca_previous_raw = previous_data.get("raw", "") or ""

        if ca_type == "redo" and ca_target:
            user_message = f"ACTION:redo_{ca_target}" + (f"|feedback={ca_feedback}" if ca_feedback else "")
        elif ca_type == "confirm":
            user_message = user_message or "ACTION:confirm"
        elif ca_type == "keep_old":
            # Restore old content to backend state
            old_content = ca_previous_raw or ca_feedback
            async def _keep_old_gen():
                if old_content and conversation_id:
                    await _restore_old_content_card(conversation_id, ca_target, old_content, context.store)
                yield context.utils.sse({"type": "done", "status": "completed"})
            return context.utils.stream_sse(_keep_old_gen())
            return context.utils.stream_sse(_keep_old_gen())

    # Handle skip_discovery (frontend "信息够了，开始策划" button)
    if body.get("skip_discovery"):
        user_message = user_message or "ACTION:confirm"

    # History action
    if action == "history":
        return await _handle_history(context, body)

    # Init LLM
    try:
        init_llm(context.env)
    except Exception as e:
        log(f"LLM init error: {e}")
        return {"status_code": 500, "body": {"error": str(e)}}

    store = context.store
    cid = conversation_id
    persistence = get_persistence()

    # Check if this is a resume (pending feedback exists)
    is_resume = has_pending(cid)
    if not is_resume:
        is_resume = await load_pending_from_store(cid, store)
    log(f"turn={'resume' if is_resume else 'kickoff'} cid={cid}")

    # Resolve deferred rollback now that pending is loaded
    if user_message == "ACTION:__rollback_deferred__":
        rollback_target = _resolve_rollback_target(cid)
        user_message = f"ACTION:{rollback_target}" if rollback_target else "ACTION:rollback_to_planning"

    # ── SSE generator ──
    async def gen():
        pending_writes: list[asyncio.Task] = []

        def fire_save(role: str, content: str, metadata: dict | None = None):
            async def _save():
                try:
                    await store.append_message(
                        conversation_id=cid, role=role,
                        content=content, metadata=metadata or {},
                    )
                except Exception as e:
                    log(f"store write failed: {e}")
            pending_writes.append(asyncio.create_task(_save()))

        try:
            yield context.utils.sse({"type": "flow_start"})

            # Emit conversation_id so frontend can track it in localStorage history
            yield context.utils.sse({"type": "conversation_id", "data": {"id": cid}})

            # Save user message
            if user_message:
                fire_save("user", user_message)

            flow = None  # Initialize for error handler access

            # ── Branch action: bypass Flow ──
            if is_resume and _is_branch_action(user_message):
                async for event in _handle_branch_action(
                    user_message, cid, persistence, locale, context
                ):
                    yield context.utils.sse(event)
                await sync_pending_to_store(cid, store)
                if pending_writes:
                    await asyncio.gather(*pending_writes, return_exceptions=True)
                # Save metadata for history (after messages are written, conversation exists)
                branch_pending = persistence.load_pending_feedback(cid)
                if branch_pending:
                    branch_state = CampaignState(**branch_pending[0])
                    await _save_conversation_metadata(cid, store, branch_state)
                yield context.utils.sse({"type": "done", "status": "completed"})
                return

            # ── Skip regeneration: confirm when next phase already has content ──
            if is_resume and user_message in ("ACTION:confirm", "ACTION:confirm"):
                can_skip = _can_skip_regeneration(cid, persistence)
                if can_skip:
                    async for event in _handle_advance_phase(cid, persistence, locale):
                        yield context.utils.sse(event)
                    await sync_pending_to_store(cid, store)
                    if pending_writes:
                        await asyncio.gather(*pending_writes, return_exceptions=True)
                    adv_pending = persistence.load_pending_feedback(cid)
                    if adv_pending:
                        adv_state = CampaignState(**adv_pending[0])
                        await _save_conversation_metadata(cid, store, adv_state)
                    yield context.utils.sse({"type": "done", "status": "completed"})
                    return

            # ── Main Flow: kickoff or resume ──
            if is_resume:
                # Append user reply to qa_history for discovery context
                pending = persistence.load_pending_feedback(cid)
                flow = MarketingCampaignFlow.from_pending(cid, persistence=persistence)

                # Inject user answer into qa_history if still in discovery
                # (but skip ACTION: messages — they're control commands, not user answers)
                if flow.state.current_phase == "discovery" and user_message and not user_message.startswith("ACTION:"):
                    flow.state.qa_history = (
                        flow.state.qa_history + f"\nUser: {user_message}"
                    ).strip()

                streaming = await _stream_resume(flow, user_message)
            else:
                # First turn: kickoff
                if not user_message:
                    yield context.utils.sse({"type": "error", "message": "Missing message"})
                    yield context.utils.sse({"type": "done", "status": "error"})
                    return

                # Safety: if user_message looks like a continuation action but we have no pending,
                # it means the session was lost. Don't silently restart — tell the user.
                if user_message.startswith("ACTION:"):
                    log(f"[ERROR] Session lost: got ACTION message but no pending state. cid={cid}")
                    yield context.utils.sse({
                        "type": "error",
                        "message": "会话状态丢失，请新建会话重试。" if locale == "zh" else "Session expired. Please start a new conversation.",
                    })
                    yield context.utils.sse({"type": "done", "status": "error"})
                    return

                flow = MarketingCampaignFlow(persistence=persistence)
                # Set campaign_brief from first message
                streaming = await flow.kickoff_async(inputs={
                    "id": cid,
                    "campaign_name": campaign_name,
                    "campaign_brief": user_message,
                    "locale": locale,
                })

            # ── Streaming loop ──
            # Record phase BEFORE streaming to detect transitions
            phase_before = flow.state.current_phase
            agent_contents: list[tuple[str, str]] = []

            yield context.utils.sse({"type": "phase_change", "phase": phase_before, "progress": _phase_progress(phase_before)})

            # For first kickoff in discovery, emit initial agent_start
            if phase_before == "discovery":
                yield context.utils.sse({"type": "agent_start", "agent": "market_analyst"})

            # ── Consume streaming output ──
            agent_contents: list[tuple[str, str]] = []
            async for event in _consume_stream(streaming, context, phase_before):
                if isinstance(event, dict) and event.get("__stream_result__"):
                    agent_contents = event["agent_contents"]
                else:
                    yield event

            # ── Post-streaming: emit structured events ──
            phase = flow.state.current_phase
            log(f"Post-streaming: phase_before={phase_before} phase_after={phase} "
                f"audience={bool(flow.state.audience_profile)} "
                f"brand={bool(flow.state.brand_creatives)} "
                f"channel={bool(flow.state.channel_plan)} "
                f"strategy={bool(flow.state.integrated_strategy)} "
                f"copy={bool(flow.state.copywriting)}")

            # Discovery phase: emit the question as a `message` event (frontend renders this as chat bubble)
            if phase_before == "discovery" and agent_contents:
                first_agent, first_content = agent_contents[0]
                # Strip [SUGGESTIONS] section for display
                display = first_content.split("[SUGGESTIONS]")[0].strip() if "[SUGGESTIONS]" in first_content else first_content
                # Also strip [READY] marker
                display = display.replace("[READY]", "").strip()
                if display:
                    yield context.utils.sse({
                        "type": "message",
                        "from": "market_analyst",
                        "content": display,
                        "phase": "discovery",
                    })

            # ── Auto-advance: if discovery_ready, immediately resume to enter planning ──
            # This avoids requiring the user to send an extra message after [READY]
            if flow.state.discovery_ready and flow.state.current_phase == "discovery":
                log("Auto-advancing from discovery to planning (discovery_ready=True)")
                streaming2 = await _stream_resume(flow, "ACTION:confirm")
                phase_before = flow.state.current_phase

                async for event in _consume_stream(streaming2, context, phase_before):
                    if isinstance(event, dict) and event.get("__stream_result__"):
                        agent_contents = event["agent_contents"]
                    else:
                        yield event

                # Update phase after auto-advance
                phase = flow.state.current_phase

            # If phase changed during streaming (e.g., discovery → planning), emit final phase
            if phase != phase_before:
                yield context.utils.sse({"type": "phase_change", "phase": phase, "progress": _phase_progress(phase)})

            # Emit card updates based on what state has
            if flow.state.audience_profile:
                yield context.utils.sse({"type": "card_update", "card": "audience", "data": {"content": flow.state.audience_profile}})
            if flow.state.brand_creatives:
                yield context.utils.sse({"type": "card_update", "card": "brand_creative", "data": {"raw": flow.state.brand_creatives}})
            if flow.state.channel_plan:
                yield context.utils.sse({"type": "card_update", "card": "channel_plan", "data": {"raw": flow.state.channel_plan}})
            if flow.state.integrated_strategy:
                yield context.utils.sse({"type": "card_update", "card": "strategy", "data": {"raw": flow.state.integrated_strategy}})
            if flow.state.copywriting:
                yield context.utils.sse({"type": "card_update", "card": "copywriting", "data": {"raw": flow.state.copywriting}})

            # Emit suggestions (discovery phase only)
            if phase == "discovery" and agent_contents:
                _, last_content = agent_contents[-1]
                suggestions = _extract_suggestions(last_content)
                if suggestions:
                    yield context.utils.sse({"type": "suggestions", "suggestions": suggestions})

            # Emit actions
            yield context.utils.sse({"type": "actions", "actions": _get_actions(phase, flow.state, locale)})

            # ── Safety: if pending was lost (cleared by resume but step failed silently),
            # restore it so the session isn't lost on next request ──
            if is_resume and flow and not has_pending(cid):
                from crewai.flow.async_feedback.types import PendingFeedbackContext
                phase_to_method = {
                    "discovery": "discovery_step",
                    "planning": "planning_step",
                    "integration": "integration_step",
                    "content": "content_step",
                    "finalize": "finalize_step",
                }
                method_name = phase_to_method.get(flow.state.current_phase, "planning_step")
                ctx = PendingFeedbackContext(
                    flow_id=cid,
                    flow_class="agents._lib.flow.MarketingCampaignFlow",
                    method_name=method_name,
                    method_output="",
                    message="(user reviews)",
                )
                persistence.save_pending_feedback(cid, ctx, flow.state.model_dump())

            # Sync to store
            await sync_pending_to_store(cid, store)
            if pending_writes:
                await asyncio.gather(*pending_writes, return_exceptions=True)
            # Save conversation metadata for history restoration (after messages are written)
            await _save_conversation_metadata(cid, store, flow.state)
            yield context.utils.sse({"type": "done", "status": "completed"})

        except Exception as e:
            log(f"stream error: {e}")
            yield context.utils.sse({"type": "error", "message": str(e)})
            # If Flow cleared pending during resume but failed before saving new pending,
            # restore the state so the user can retry (not lose entire session)
            if is_resume and not has_pending(cid):
                log(f"Restoring pending state after error for cid={cid}")
                from crewai.flow.async_feedback.types import PendingFeedbackContext
                try:
                    phase = flow.state.current_phase if flow else phase_before
                    phase_to_method = {
                        "discovery": "discovery_step",
                        "planning": "planning_step",
                        "integration": "integration_step",
                        "content": "content_step",
                        "finalize": "finalize_step",
                    }
                    method_name = phase_to_method.get(phase, "planning_step")
                    state_data = flow.state.model_dump() if flow else {}
                    if state_data:
                        ctx = PendingFeedbackContext(
                            flow_id=cid,
                            flow_class="agents._lib.flow.MarketingCampaignFlow",
                            method_name=method_name,
                            method_output="",
                            message="(user reviews)",
                        )
                        persistence.save_pending_feedback(cid, ctx, state_data)
                except Exception as restore_err:
                    log(f"Failed to restore pending: {restore_err}")
            await sync_pending_to_store(cid, store)
            if pending_writes:
                await asyncio.gather(*pending_writes, return_exceptions=True)
            yield context.utils.sse({"type": "done", "status": "error"})

    return context.utils.stream_sse(gen())


# ─── Branch action handler ───────────────────────────────────────────

async def _handle_branch_action(message, cid, persistence, locale, context):
    """Handle redo/rollback actions — direct crew calls, bypass Flow."""
    action, feedback = _parse_action(message)
    locale_instruction = "Chinese (中文)" if locale == "zh" else "English"

    # Load current state from persistence
    pending = persistence.load_pending_feedback(cid)
    if not pending:
        yield {"type": "error", "message": "No pending state found"}
        return
    state_data, _ = pending
    state = CampaignState(**state_data)

    inputs = {
        "campaign_name": state.campaign_name,
        "campaign_brief": state.campaign_brief + (f"\n\nFeedback: {feedback}" if feedback else ""),
        "audience_profile": state.audience_profile,
        "market_insights": state.market_insights,
        "locale_instruction": locale_instruction,
    }

    if action == "redo_brand":
        yield {"type": "phase_change", "phase": "planning"}
        yield {"type": "agent_start", "agent": "Brand & Creative Director"}
        result = BrandCreativeCrew().crew().kickoff(inputs=inputs)
        state.brand_creatives = _crew_text(result)
        yield {"type": "agent_end", "agent": "Brand & Creative Director"}
        yield {"type": "card_update", "card": "brand_creative", "data": {"raw": state.brand_creatives}}
        # Mark downstream phases as needing regeneration
        state.invalidated_phases = list(set(getattr(state, 'invalidated_phases', []) + ["integration", "content", "finalize"]))

    elif action == "redo_channel":
        yield {"type": "phase_change", "phase": "planning"}
        yield {"type": "agent_start", "agent": "Channel & Media Planner"}
        result = ChannelPlanningCrew().crew().kickoff(inputs=inputs)
        state.channel_plan = _crew_text(result)
        yield {"type": "agent_end", "agent": "Channel & Media Planner"}
        yield {"type": "card_update", "card": "channel_plan", "data": {"raw": state.channel_plan}}
        # Mark downstream phases as needing regeneration
        state.invalidated_phases = list(set(getattr(state, 'invalidated_phases', []) + ["integration", "content", "finalize"]))

    elif action == "rollback_to_planning":
        state.current_phase = "planning"
        yield {"type": "phase_change", "phase": "planning"}
        yield {"type": "card_update", "card": "brand_creative", "data": {"raw": state.brand_creatives}}
        yield {"type": "card_update", "card": "channel_plan", "data": {"raw": state.channel_plan}}

    elif action == "rollback_to_integration":
        state.current_phase = "integration"
        yield {"type": "phase_change", "phase": "integration"}
        yield {"type": "card_update", "card": "strategy", "data": {"raw": state.integrated_strategy}}

    elif action == "rollback_to_content":
        state.current_phase = "content"
        yield {"type": "phase_change", "phase": "content"}
        yield {"type": "card_update", "card": "copywriting", "data": {"raw": state.copywriting}}

    # Save updated state back to persistence with correct method_name
    # so Flow resumes from the right step on next interaction
    from crewai.flow.async_feedback.types import PendingFeedbackContext
    phase_to_method = {
        "planning": "planning_step",
        "integration": "integration_step",
        "content": "content_step",
        "finalize": "finalize_step",
    }
    method_name = phase_to_method.get(state.current_phase, "planning_step")
    pending_ctx = PendingFeedbackContext(
        flow_id=cid,
        flow_class="agents._lib.flow.MarketingCampaignFlow",
        method_name=method_name,
        method_output="",
        message="(user reviews)",
    )
    persistence.save_pending_feedback(cid, pending_ctx, state.model_dump())

    yield {"type": "actions", "actions": _get_actions(state.current_phase, state, locale)}


# ─── History handler ─────────────────────────────────────────────────

async def _handle_history(context, body):
    """Return conversation history from context.store."""
    conversation_id = getattr(context, "conversation_id", "") or body.get("conversation_id", "")
    if not conversation_id:
        return {"conversation_id": "", "chat_history": [], "current_phase": "start"}

    try:
        store = context.store
        messages = await store.get_messages(
            conversation_id=conversation_id, limit=100, order="asc"
        )
        chat_history = []
        for m in messages:
            meta_data = m.metadata or {}
            if meta_data.get("type") == "init":
                continue
            # Skip ACTION: messages (internal control commands, not user-facing)
            if m.content and m.content.startswith("ACTION:"):
                continue
            agent = meta_data.get("agent", m.role)
            chat_history.append({"role": agent, "content": m.content})

        # Try to determine phase and cards from persistence
        persistence = get_persistence()
        current_phase = "discovery" if chat_history else "start"
        cards = {}

        # First try in-memory pending
        pending = persistence.load_pending_feedback(conversation_id)
        if not pending:
            # Try to load from store
            await load_pending_from_store(conversation_id, store)
            pending = persistence.load_pending_feedback(conversation_id)

        if pending:
            state_data = pending[0]
            current_phase = state_data.get("current_phase", current_phase)
            # Build cards from state
            if state_data.get("brand_creatives"):
                cards["brand_creative"] = {"raw": state_data["brand_creatives"]}
            if state_data.get("channel_plan"):
                cards["channel_plan"] = {"raw": state_data["channel_plan"]}
            if state_data.get("integrated_strategy"):
                cards["strategy"] = {"raw": state_data["integrated_strategy"], "content": state_data["integrated_strategy"]}
            if state_data.get("copywriting"):
                cards["copywriting"] = {"raw": state_data["copywriting"]}
            if state_data.get("audience_profile"):
                cards["audience"] = {"content": state_data["audience_profile"]}

        # Fallback: try conversation metadata
        if not cards:
            try:
                meta = await store.get_conversation(conversation_id=conversation_id)
                if meta and meta.metadata:
                    current_phase = meta.metadata.get("current_phase", current_phase)
                    cards = meta.metadata.get("cards", {})
            except Exception:
                pass

        return {
            "conversation_id": conversation_id,
            "chat_history": chat_history,
            "current_phase": current_phase,
            "cards": cards,
        }
    except Exception:
        return {"conversation_id": conversation_id, "chat_history": [], "current_phase": "start"}


# ─── Utility functions ───────────────────────────────────────────────

async def _restore_old_content_card(cid: str, target: str, old_content: str, store) -> None:
    """Restore old content for a specific card (brand/channel) to persistence."""
    persistence = get_persistence()

    if not has_pending(cid):
        await load_pending_from_store(cid, store)

    pending = persistence.load_pending_feedback(cid)
    if not pending:
        return
    state_data, ctx = pending

    if target == "brand":
        state_data["brand_creatives"] = old_content
    elif target == "channel":
        state_data["channel_plan"] = old_content

    persistence.save_pending_feedback(cid, ctx, state_data)
    await sync_pending_to_store(cid, store)


async def _restore_old_content(cid: str, old_content: str, phase: str, store) -> None:
    """Restore old content to persistence when user chooses 'keep old' in compare mode."""
    persistence = get_persistence()

    if not has_pending(cid):
        await load_pending_from_store(cid, store)

    pending = persistence.load_pending_feedback(cid)
    if not pending:
        return
    state_data, ctx = pending

    current_phase = state_data.get("current_phase", phase)
    if current_phase == "integration":
        state_data["integrated_strategy"] = old_content
    elif current_phase == "content":
        state_data["copywriting"] = old_content

    persistence.save_pending_feedback(cid, ctx, state_data)
    await sync_pending_to_store(cid, store)


def _phase_progress(phase: str) -> int:
    return {
        "discovery": 10,
        "planning": 30,
        "integration": 55,
        "content": 75,
        "finalize": 95,
    }.get(phase, 0)


def _extract_suggestions(text: str) -> list[str]:
    """Extract [SUGGESTIONS] section from discovery output."""
    if "[SUGGESTIONS]" not in text:
        return []
    parts = text.split("[SUGGESTIONS]", 1)
    if len(parts) < 2:
        return []
    suggestions = []
    for line in parts[1].strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            suggestions.append(line[2:].strip())
        elif line:
            suggestions.append(line)
    return suggestions[:3]


def _get_actions(phase: str, state, locale: str) -> list[dict]:
    """Generate available actions for current phase."""
    zh = locale == "zh"
    actions = []

    if phase == "discovery":
        pass  # Just reply to continue

    elif phase == "planning":
        actions = [
            {"id": "ACTION:confirm", "label": "确认方案，继续" if zh else "Confirm & Continue"},
            {"id": "ACTION:redo_brand", "label": "重新生成品牌创意" if zh else "Redo Brand Creative"},
            {"id": "ACTION:redo_channel", "label": "重新生成渠道策略" if zh else "Redo Channel Strategy"},
        ]

    elif phase == "integration":
        actions = [
            {"id": "ACTION:confirm", "label": "确认，继续" if zh else "Confirm & Continue"},
            {"id": "ACTION:rollback_to_planning", "label": "返回方案策划" if zh else "Back to Planning"},
        ]

    elif phase == "content":
        actions = [
            {"id": "ACTION:confirm", "label": "确认，完成" if zh else "Confirm & Finish"},
            {"id": "ACTION:rollback_to_integration", "label": "返回策略整合" if zh else "Back to Integration"},
        ]

    elif phase == "finalize":
        actions = [
            {"id": "ACTION:generate_document", "label": "生成完整方案" if zh else "Generate Full Plan"},
        ]
        if state.integrated_strategy and len(state.integrated_strategy) > 500:
            actions.append(
                {"id": "ACTION:revise_document", "label": "修改方案" if zh else "Revise Plan"}
            )

    return actions


def _agent_to_lane(agent_role: str) -> str:
    """Map agent role to parallel lane name (for planning phase UI)."""
    agent_id = _normalize_agent(agent_role)
    if agent_id == "brand_creative_director":
        return "brand"
    if agent_id == "channel_planner":
        return "channel"
    return ""


def _resolve_rollback_target(cid: str) -> str:
    """Map generic 'rollback' to the correct phase-specific action based on current state."""
    persistence = get_persistence()
    pending = persistence.load_pending_feedback(cid)
    if not pending:
        return ""
    state_data = pending[0]
    current_phase = state_data.get("current_phase", "")
    # Map: current phase → which phase to roll back to
    mapping = {
        "integration": "rollback_to_planning",
        "content": "rollback_to_integration",
        "finalize": "rollback_to_content",
    }
    return mapping.get(current_phase, "")


def _can_skip_regeneration(cid: str, persistence) -> bool:
    """Check if we can skip regeneration on confirm (next phase has valid content)."""
    pending = persistence.load_pending_feedback(cid)
    if not pending:
        return False
    state_data = pending[0]
    current_phase = state_data.get("current_phase", "")
    invalidated = state_data.get("invalidated_phases", [])

    # Determine next phase
    next_phase_map = {
        "planning": "integration",
        "integration": "content",
        "content": "finalize",
    }
    next_phase = next_phase_map.get(current_phase, "")
    if not next_phase:
        return False

    # Check if next phase is invalidated
    if next_phase in invalidated:
        return False

    # Check if next phase already has content
    content_check = {
        "integration": bool(state_data.get("integrated_strategy")),
        "content": bool(state_data.get("copywriting")),
        "finalize": bool(state_data.get("copywriting")),  # finalize just needs content to exist
    }
    return content_check.get(next_phase, False)


async def _handle_advance_phase(cid, persistence, locale):
    """Advance to next phase without regeneration — emit existing content."""
    from crewai.flow.async_feedback.types import PendingFeedbackContext

    pending = persistence.load_pending_feedback(cid)
    if not pending:
        yield {"type": "error", "message": "No pending state found"}
        return
    state_data, _ = pending
    state = CampaignState(**state_data)

    current_phase = state.current_phase
    next_phase_map = {
        "planning": "integration",
        "integration": "content",
        "content": "finalize",
    }
    next_phase = next_phase_map.get(current_phase, "finalize")

    # Advance state
    state.current_phase = next_phase
    # Remove this phase from invalidated list (if it was there)
    invalidated = getattr(state, 'invalidated_phases', [])
    state.invalidated_phases = [p for p in invalidated if p != next_phase]

    yield {"type": "phase_change", "phase": next_phase, "progress": _phase_progress(next_phase)}

    # Emit existing card data for the phase
    if next_phase == "integration" and state.integrated_strategy:
        yield {"type": "card_update", "card": "strategy", "data": {"raw": state.integrated_strategy, "content": state.integrated_strategy}}
    elif next_phase == "content" and state.copywriting:
        yield {"type": "card_update", "card": "copywriting", "data": {"raw": state.copywriting}}
    elif next_phase == "finalize":
        pass  # Finalize phase doesn't auto-generate

    # Save updated state
    phase_to_method = {
        "planning": "planning_step",
        "integration": "integration_step",
        "content": "content_step",
        "finalize": "finalize_step",
    }
    method_name = phase_to_method.get(next_phase, "finalize_step")
    pending_ctx = PendingFeedbackContext(
        flow_id=cid,
        flow_class="agents._lib.flow.MarketingCampaignFlow",
        method_name=method_name,
        method_output="",
        message="(user reviews)",
    )
    persistence.save_pending_feedback(cid, pending_ctx, state.model_dump())
    yield {"type": "actions", "actions": _get_actions(next_phase, state, locale)}


async def _save_conversation_metadata(cid: str, store, state) -> None:
    """Save current_phase and cards to conversation metadata for /history endpoint."""
    cards = {}
    if state.brand_creatives:
        cards["brand_creative"] = {"raw": state.brand_creatives}
    if state.channel_plan:
        cards["channel_plan"] = {"raw": state.channel_plan}
    if state.integrated_strategy:
        cards["strategy"] = {"raw": state.integrated_strategy, "content": state.integrated_strategy}
    if state.copywriting:
        cards["copywriting"] = {"raw": state.copywriting}
    if state.audience_profile:
        cards["audience"] = {"content": state.audience_profile}

    try:
        # Ensure conversation exists before updating metadata
        try:
            await store.get_conversation(conversation_id=cid)
        except Exception:
            await store.append_message(
                conversation_id=cid,
                role="system",
                content=f"Campaign: {getattr(state, 'campaign_name', '')}",
                metadata={"type": "init"},
            )
        await store.update_conversation(
            conversation_id=cid,
            metadata={
                "current_phase": state.current_phase,
                "cards": cards,
                "campaign_name": getattr(state, "campaign_name", ""),
            },
        )
    except Exception as e:
        log(f"save conversation metadata failed: {e}")


def _normalize_agent(agent_role: str) -> str:
    """Map CrewAI agent role string to frontend agent_id.

    Frontend expects: market_analyst, brand_creative_director, channel_planner,
    chief_strategist, copywriter.
    CrewAI sends the 'role' field from agents.yaml which is a human-readable string.
    """
    if not agent_role:
        return ""
    lower = agent_role.lower().strip()
    if "market" in lower and ("analyst" in lower or "research" in lower):
        return "market_analyst"
    if "brand" in lower or "creative director" in lower:
        return "brand_creative_director"
    if "channel" in lower or "media" in lower:
        return "channel_planner"
    if "strategist" in lower or "chief" in lower:
        return "chief_strategist"
    if "copywriter" in lower or "copy" in lower:
        return "copywriter"
    return agent_role
