import { useReducer, useCallback, useState, useEffect } from "react"
import type { AppState, AppAction, Phase, Locale, ChatMessage } from "./types"
import { useSSE } from "./hooks/useSSE"
import { saveHistory } from "./hooks/useHistory"
import { t, setLocale, getLocale } from "./i18n"
import Header from "./components/Header"
import PhaseProgress from "./components/PhaseProgress"
import StatusBar from "./components/StatusBar"
import InputBar from "./components/InputBar"
import StartPanel from "./components/StartPanel"
import HistoryPanel from "./components/HistoryPanel"
import DiscoveryView from "./components/views/DiscoveryView"
import PlanningView from "./components/views/PlanningView"
import IntegrationView from "./components/views/IntegrationView"
import ContentView from "./components/views/ContentView"
import FinalizeView from "./components/views/FinalizeView"
import { exportCampaignMarkdown, downloadMarkdown } from "./utils/export"

const PHASE_ORDER: Phase[] = ["start", "discovery", "planning", "integration", "content", "finalize"]

function getProgress(phase: Phase): number {
  const idx = PHASE_ORDER.indexOf(phase)
  if (idx <= 0) return 0
  return Math.round((idx / (PHASE_ORDER.length - 2)) * 100)
}

const initialState: AppState = {
  phase: "start",
  progress: 0,
  locale: getLocale(),
  conversationId: "",
  streaming: false,
  activeAgents: [],
  pendingActions: [],
  statusMessage: "",
  cards: {
    audience: null,
    brand_creative: null,
    channel_plan: null,
    strategy: null,
    copywriting: null,
  },
  messages: [],
  parallelActive: false,
  suggestions: [],
  finalDocument: "",
  reconciliationNotes: "",
}

function reducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "RESET":
      return { ...initialState, locale: state.locale }

    case "SET_LOCALE":
      return { ...state, locale: action.locale }

    case "SET_CONVERSATION_ID":
      return { ...state, conversationId: action.id }

    case "PHASE_CHANGE":
      return {
        ...state,
        phase: action.phase,
        progress: action.progress || getProgress(action.phase),
        pendingActions: [],
        statusMessage: "",
      }

    case "AGENT_START":
      return {
        ...state,
        activeAgents: [...state.activeAgents, { agent: action.agent, lane: action.lane }],
        // Clear finalDocument when chief_strategist starts in finalize (regenerating document)
        ...(state.phase === "finalize" && action.agent === "chief_strategist" ? { finalDocument: "" } : {}),
      }

    case "CHUNK": {
      const content = action.content

      // discovery 阶段不处理流式 chunk
      if (state.phase === "discovery") {
        return state
      }

      // finalize 阶段：chief_strategist 的 chunk 写入 finalDocument
      if (state.phase === "finalize" && action.agent === "chief_strategist") {
        return { ...state, finalDocument: state.finalDocument + content }
      }

      // 追加到 messages
      const lastMsg = state.messages[state.messages.length - 1]
      let newMessages = state.messages
      if (lastMsg && lastMsg.role === action.agent && lastMsg.phase === state.phase) {
        newMessages = [...state.messages]
        newMessages[newMessages.length - 1] = { ...lastMsg, content: lastMsg.content + content }
      } else {
        newMessages = [...state.messages, { role: action.agent, content, phase: state.phase }]
      }

      // 同时更新 cards 数据（让卡片组件流式渲染）
      let newCards = state.cards
      const agentCardMap: Record<string, string> = {
        brand_creative_director: "brand_creative",
        channel_planner: "channel_plan",
        chief_strategist: "strategy",
        copywriter: "copywriting",
      }
      const cardKey = agentCardMap[action.agent]
      if (cardKey) {
        const existing = (state.cards[cardKey as keyof typeof state.cards] as Record<string, unknown>) || {}
        let newRaw = ((existing.raw as string) || "") + content
        // 清理累积内容中的完整 HTML 标签（解决标签被分割到多个 chunk 的问题）
        newRaw = newRaw.replace(/<br\s*\/?>/gi, "\n").replace(/<[^>]+>/g, "")
        newCards = { ...state.cards, [cardKey]: { raw: newRaw } }
      }

      return { ...state, messages: newMessages, cards: newCards }
    }

    case "AGENT_END":
      return {
        ...state,
        activeAgents: state.activeAgents.filter(
          (a) => !(a.agent === action.agent && a.lane === action.lane)
        ),
      }

    case "CLEAR_CARD":
      return {
        ...state,
        cards: { ...state.cards, [action.card]: null },
      }

    case "CARD_UPDATE": {
      // 递归清理 data 中所有字符串的 HTML 标签
      const cleanHtml = (obj: unknown): unknown => {
        if (typeof obj === "string") {
          return obj.replace(/<br\s*\/?>/gi, "\n").replace(/<[^>]+>/g, "")
        }
        if (Array.isArray(obj)) return obj.map(cleanHtml)
        if (obj && typeof obj === "object") {
          const result: Record<string, unknown> = {}
          for (const [k, v] of Object.entries(obj)) {
            result[k] = cleanHtml(v)
          }
          return result
        }
        return obj
      }
      return {
        ...state,
        cards: { ...state.cards, [action.card]: cleanHtml(action.data) as Record<string, unknown> },
      }
    }

    case "ACTIONS":
      return { ...state, pendingActions: action.actions }

    case "STATUS":
      return { ...state, statusMessage: action.message }

    case "CLEAR_FINAL_DOC":
      return { ...state, finalDocument: "", reconciliationNotes: "" }

    case "SET_RECONCILIATION":
      return { ...state, reconciliationNotes: action.notes }

    case "PARALLEL_START":
      return { ...state, parallelActive: true }

    case "PARALLEL_END":
      return { ...state, parallelActive: false }

    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.message] }

    case "STREAMING_START":
      return { ...state, streaming: true, suggestions: [] }

    case "STREAMING_END":
      return { ...state, streaming: false, statusMessage: state.statusMessage.startsWith("Error") ? state.statusMessage : "" }

    case "DONE":
      return { ...state, streaming: false, statusMessage: state.statusMessage.startsWith("Error") ? state.statusMessage : "" }

    case "ERROR":
      return { ...state, streaming: false, statusMessage: `Error: ${action.message}` }

    case "RESTORE":
      return { ...state, ...action.state }

    case "SUGGESTIONS":
      return { ...state, suggestions: action.suggestions }

    default:
      return state
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const { send, abort, loadHistory: loadHistoryFromServer, setConversationId } = useSSE(dispatch)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [prefillValue, setPrefillValue] = useState("")

  const handleLocaleChange = useCallback((locale: Locale) => {
    setLocale(locale)
    dispatch({ type: "SET_LOCALE", locale })
  }, [])

  const [campaignNameRef, setCampaignNameRef] = useState("")

  // 在获得 conversationId 后保存历史
  useEffect(() => {
    if (state.conversationId && campaignNameRef && state.phase !== "start") {
      saveHistory(state.conversationId, campaignNameRef, state.phase)
    }
  }, [state.conversationId, campaignNameRef, state.phase])

  const handleStart = useCallback((name: string, brief: string) => {
    setCampaignNameRef(name)
    send({
      campaign_name: name,
      campaign_brief: brief,
      locale: state.locale,
    })
  }, [send, state.locale])

  const handleSelectHistory = useCallback(async (id: string) => {
    dispatch({ type: "STREAMING_START" })
    await loadHistoryFromServer(id)
    dispatch({ type: "STREAMING_END" })
  }, [loadHistoryFromServer])

  const handleSendMessage = useCallback((message: string) => {
    const userMsg: ChatMessage = { role: "user", content: message, phase: state.phase }
    dispatch({ type: "ADD_MESSAGE", message: userMsg })
    send({ message, locale: state.locale })
    setPrefillValue("")
  }, [send, state.phase, state.locale])

  const handleCardAction = useCallback((target: "brand" | "channel", type: "confirm" | "redo" | "keep_old", options?: { selected_index?: number; feedback?: string; previous_data?: Record<string, unknown> }) => {
    // redo 时清空对应卡片，让流式从空开始
    if (type === "redo") {
      const cardKey = target === "brand" ? "brand_creative" : "channel_plan"
      dispatch({ type: "CLEAR_CARD", card: cardKey })
    }
    send({
      card_action: { target, type, ...options },
      locale: state.locale,
    })
  }, [send, state.locale])

  const handlePhaseAction = useCallback((type: "confirm" | "redo" | "rollback" | "keep_old", feedback?: string) => {
    // redo 时清空对应卡片
    if (type === "redo") {
      if (state.phase === "integration") {
        dispatch({ type: "CLEAR_CARD", card: "strategy" })
      } else if (state.phase === "content") {
        dispatch({ type: "CLEAR_CARD", card: "copywriting" })
      }
    }
    // keep_old 时恢复前端卡片显示
    if (type === "keep_old" && feedback) {
      if (state.phase === "integration") {
        dispatch({ type: "CARD_UPDATE", card: "strategy", data: { raw: feedback } })
      } else if (state.phase === "content") {
        dispatch({ type: "CARD_UPDATE", card: "copywriting", data: { raw: feedback } })
      }
    }
    send({
      phase_action: { type, feedback },
      locale: state.locale,
    })
  }, [send, state.locale, state.phase])

  const handleSkipDiscovery = useCallback(() => {
    send({ skip_discovery: true, locale: state.locale })
  }, [send, state.locale])

  const handleNew = useCallback(() => {
    abort()
    setConversationId("")
    setCampaignNameRef("")
    dispatch({ type: "RESET" })
  }, [abort, setConversationId])

  const renderView = () => {
    switch (state.phase) {
      case "start":
        return <StartPanel onStart={handleStart} disabled={state.streaming} />
      case "discovery":
        return (
          <DiscoveryView
            messages={state.messages}
            streaming={state.streaming}
            activeAgents={state.activeAgents}
            onSkip={handleSkipDiscovery}
            suggestions={state.suggestions}
            onSuggestionClick={(s) => setPrefillValue(s)}
          />
        )
      case "planning":
        return (
          <PlanningView
            cards={state.cards}
            streaming={state.streaming}
            activeAgents={state.activeAgents}
            parallelActive={state.parallelActive}
            onCardAction={handleCardAction}
            onRestoreCard={(card, data) => dispatch({ type: "CARD_UPDATE", card, data })}
            onNext={() => send({ phase_action: { type: "confirm" }, locale: state.locale })}
          />
        )
      case "integration":
        return (
          <IntegrationView
            cards={state.cards}
            streaming={state.streaming}
            activeAgents={state.activeAgents}
            onAction={handlePhaseAction}
          />
        )
      case "content":
        return (
          <ContentView
            cards={state.cards}
            streaming={state.streaming}
            activeAgents={state.activeAgents}
            onAction={handlePhaseAction}
          />
        )
      case "finalize":
        return (
          <FinalizeView
            cards={state.cards}
            streaming={state.streaming}
            activeAgents={state.activeAgents}
            finalDocument={state.finalDocument}
            onGenerateDocument={() => {
              dispatch({ type: "CLEAR_FINAL_DOC" })
              send({ phase_action: { type: "confirm" }, locale: state.locale })
            }}
            onEditDocument={(fb) => {
              dispatch({ type: "CLEAR_FINAL_DOC" })
              send({ iteration_feedback: fb, locale: state.locale })
            }}
            onGoBackToContent={() => send({ phase_action: { type: "rollback" }, locale: state.locale })}
            onGoBackToOverview={() => dispatch({ type: "CLEAR_FINAL_DOC" })}
            onExportCards={() => downloadMarkdown(exportCampaignMarkdown(state.cards), `campaign-cards-${Date.now()}.md`)}
            onExportDocument={() => downloadMarkdown(state.finalDocument, `campaign-plan-${Date.now()}.md`)}
          />
        )
      default:
        return null
    }
  }

  const showInputBar = state.phase === "discovery"

  return (
    <div className="app-shell flex flex-col bg-[var(--color-bg)]">
      <Header
        locale={state.locale}
        onLocaleChange={handleLocaleChange}
      />

      {state.phase !== "start" ? (
        <PhaseProgress
          phase={state.phase}
          progress={state.progress}
          onNew={handleNew}
          onHistory={() => setHistoryOpen(true)}
        />
      ) : (
        <div className="flex justify-end px-6 py-2 gap-2">
          <button onClick={handleNew} className="btn btn-ghost text-xs cursor-pointer px-2 py-1">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            {t("app.new")}
          </button>
          <button onClick={() => setHistoryOpen(true)} className="btn btn-ghost text-xs cursor-pointer px-2 py-1">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {t("app.history")}
          </button>
        </div>
      )}

      {/* 全局 streaming 指示条 */}
      {state.streaming && (
        <div className="h-0.5 bg-[var(--color-primary)]/20 overflow-hidden">
          <div className="h-full bg-[var(--color-primary)] animate-[progress_2s_ease-in-out_infinite] w-1/3" />
        </div>
      )}

      <main className="flex-1 overflow-y-auto px-6 py-4">
        {renderView()}
      </main>

      {state.phase !== "start" && (state.statusMessage || state.streaming) && (
        <StatusBar message={state.statusMessage || (state.streaming ? (t("status.generating") || "处理中...") : "")} />
      )}

      {showInputBar && (
        <InputBar
          onSend={handleSendMessage}
          disabled={state.streaming}
          placeholder={t("input.placeholder")}
          prefill={prefillValue}
        />
      )}

      <HistoryPanel
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        onSelect={handleSelectHistory}
        currentId={state.conversationId}
      />
    </div>
  )
}
