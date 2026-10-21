# What makes Siri Siri — macOS 27 (Darwin 27.0.0), static reverse-engineering pass

Method: read-only static inspection of the shipped OS on this Mac — launchd plists, Info.plists,
code-signing entitlements, `dyld_info -exports/-linked_dylibs/-section __TEXT __cstring` against the
dyld shared cache, Swift demangling, and shipped resource/asset files. No decompilation, no runtime
tracing, no REA (not connected). Evidence = **O** observed in a shipped file, **I** inference.

Raw evidence (inventory JSON, string tables) was produced in a temporary scratchpad and not kept;
re-run with `dyld_info -exports` / `-section __TEXT __cstring` against the shared cache to regenerate.

---

## 0. The shape of it

Siri is ~292 frameworks (808k exported symbols, ~30k Swift types, ~10.6k ObjC classes) and ~17
launchd services. There are **two Siris shipped side by side**:

| | Legacy Siri ("SiriX") | New LLM Siri ("Campo" / IntelligenceFlow) |
|---|---|---|
| UI app | `/System/Library/CoreServices/Siri.app` (com.apple.Siri) | `/System/Applications/Siri AI.app` (com.apple.campo) **O** |
| Brain | `assistantd` + `assistant_cdmd` (Continuous Dialog Manager NLU) + SiriKitFlow | `intelligenceflowd` — AgenticPlanner LLM tool-loop **O** |
| Gate | always | launchd `#IfFeatureFlagDisabled IntelligenceFlow/Campo` → Disabled **O** |
| Relationship | exposed to the planner as one tool, `siri_x` **O** | calls into legacy for Settings/Home/Car/Health/etc. |

## 1. Activation — hearing "Siri" (corespeechd, VoiceTrigger, CoreSpeechExclave)

Processes: `corespeechd` (KeepAlive, Interactive), `corespeechd_system` (user `_corespeechd`). **O**
Services: voicetriggerservice, endpointer.service, attending.service, speechprofileadmin, voiceprofilemanager. **O**

Wake-word cascade (asset `AssetsV2/com_apple_MobileAsset_VoiceTriggerAssetsASMac/.../corespeech.json`, all **O**):
1. **First pass** — tiny always-on model (`config_1st.txt`, threshold −2.91; engine header "Novauris"),
   signed `rtblob_*.bin` for the always-on processor; `CoreSpeechExclave` runs it in a secure exclave
   (`AOPVoiceTriggerResult`, `CSSecureAOEAudioProvidingProxy`).
2. **Second pass** — keyword spotting with a CTC phoneme model `megatron_hsjs_v2_8bit.mlmodelc`, 40-mel
   fbank @16 kHz. Keywords: `HS1/HS2` = "hey s(EE|IH)ree", `JS1/JS2` = "s(EE|IH)ree" ("just Siri").
   Per-phrase thresholds (HS −0.18, Siri 1.1 — "Siri" alone needs a much higher score), plus a
   **2nd-chance threshold** and 0.3 s pre/post audio padding.
3. **Speaker recognition** — SAT thresholds (HS 0.592 / Siri 0.573), implicit profile training on
   accepted triggers (up to 40 utterances), multi-user score thresholds.
4. **False-trigger mitigation** — `mitigation/`: acoustic FTM `aftm` (1.92 s checker), `sdsd`
   mega_model_v7 fusing lrnn score + acoustic FTM + input origin, and `nlda` — an on-device text
   classifier over the ASR text with regex pre-mapping (`^Hey Siri\W*$` → ACCEPT, otherwise
   `[NO_TRIGGER]`, etc.).
5. **Attending / Liminal** — `AttSiri*` configs: follow-ups without a wake word, gated by start-of-speech
   (150 ms), 700 ms pre-roll, speaker ID, and the same mitigation models. `SiriTurnTakingManager`
   (`TTMitigationDecision`) decides if follow-up speech was meant for Siri.
6. **Cross-device arbitration** — `SiriCrossDeviceArbitration`: BLE "goodness score" election so only
   one device answers (`SCDAGoodnessScoreEvaluator`, `SCDAElectionWindow`, perceptual audio hash).

## 2. Speech → text (CoreEmbeddedSpeechRecognition, LocalSpeechRecognitionBridge)

- On-device ASR (`UAF_Speech_AutomaticSpeechRecognition`, 395 MB) **O**, server fallback class
  `CESRServerSpeechRecognizer` **O**.
- **Personalized vocabulary**: "Euclid" speech profiles built from contacts, app names, HomeKit
  entities, media, Find My, podcasts, custom terms (`CESRRawSpeechProfileItemConverter*`) + a
  phonetic embedder; ASR n-best is re-ranked against app entities (`CESRAppEntityRanker`,
  `ASREntityRanker`, `ASRRankingModel.mlmodelc` in the planner). **O**
- Planner config flags: `asrCorrectionsEntityRetrieval`, `asrCorrectionsNBest`, `asrCorrectionsSNR`. **O**

## 3. Understanding, context and memory

- **Legacy NLU**: `assistant_cdmd` (CDM) + `SiriNaturalLanguageParsing` (SSU matcher, ITFM classifier,
  server "Pommes" classifier) + `SiriNLUTypes` (protobuf `Siri_Nlu_External_*`, `NLU_ROUTER`). **O**
- **Reference resolution** (`siriknowledged`, `SiriReferenceResolution`): "that", "him", "this" are
  resolved from a salience-ranked entity pool fed by pullers for foreground app, on-screen UI
  context, notifications, now playing, nearby people, RunningBoard, voice shortcuts. **O**
- **On-screen awareness** (`intelligencecontextd`, `IntelligenceFlowContext`): `OnScreenContent`,
  `OnScreenPerson/PhoneNumber/PostalAddress/UriLink`, UI element hierarchy fetch, entity detectors
  (BERT, NSDataDetector, SiriEntityMatcher). **O**
- **Personal knowledge graph** (`intelligenceplatformd`, `knowledgeconstructiond`): entity graph of
  people, events, places, activities (`GDGraph*`), entity resolution, "Knosis" indexes — Siri AI's
  entitlement reads `stableGraphIndex`, `eventGraphIndex`, `liveGlobalKnowledgeIndex`,
  `staticGlobalKnowledgeIndex`, `ontologyIndex`, and views `siriRemembers`, `entitySummary`,
  `peopleSubgraph`. **O**
- **Habits** (`siriinferenced`: SiriRemembers + SiriSignals): ~120 signals such as app-launch counts,
  domain→contact affinity, recency, hour-of-day, CarPlay, workout; used for app/contact choice
  ("call Mom" → which Mom, which app). **O**
- **Private learning** (`SiriPrivateLearningInference`): offline session stitching, ground-truth inference
  from what the user did next (e.g. re-asked, corrected), adaptive contact/media suggestions. **O**

## 4. The LLM brain — intelligenceflowd / AgenticPlanner

Services: `intelligenceflow.orchestrator`, `.toolbox`, `.querydecoration`, `.snippet-streaming`,
`.imageretrieval`, `.activation.prewarm`, `siri.orchestration.prescribedaction`. **O**

**Models** (string literals in IntelligenceFlowPlannerRuntime and related frameworks, **O**; sizes inferred from the names, **I**):
- Planner on-device: `com.apple.fm.language.instruct_3b.lw_planner_v1` (path "ODM")
- Planner in Private Cloud Compute: `instruct_server_v1/v2.lw_planner_v1…v5`, `lw_planner_upgradeable` (path "PCC")
- Guards (300M): `instruct_300m.action_validator`, `instruct_300m.ipi_classifier` (indirect prompt injection), `instruct_300m.safety`, plus `prompt_injection`
- Misc: `instruct_server_v2.pcc_interact`, `conversation_title_summarization`
- Routing: `'odm', 'pcc', or nil`; `SiriPCCAgentRouter` with "sticky Pro route" + `contextHeadroomFraction` **O**
- Agent framework (`GenerativeAgents`): before/after model, tool, and agent callbacks, `TransferToAgent`,
  `AgentRoutingMode`, `ModelContextProtocol.ToolDefinition`, `OpenAICompatibleModel`,
  `OpenAICodableFormat.ChatResponse` — a multi-agent runtime that can speak OpenAI wire format. **O**

**Request flow** (**I**, assembled from the types and strings above):
1. **Skimmer fast path**: `EnigmaUtteranceClassifier` matches 622 regexes / 153 actions (en_US
   `enigma_patterns/en_US/Enigma.json`) such as `set_timer`, `make_call`, `get_weather(...)`,
   `math_calculate(tmpl='{a} * {b}')`, `siri_x_mini_tool(tool='home_device_control')`. A hit runs a
   deterministic flow with no LLM. Misses, or a failed step, fall back to the planner ("Skimmer fallback to model"). **O**
2. **Query decoration / tool retrieval**: staged tool selection (`stage1And2ToolMinimum`,
   `stage3ToolMinimum`, `semanticToolsForRankerCount`, `spanMatchedToolsForRankerCount`,
   `toolSelectionTargetToolCount`) narrows a catalog of 2,833 App Intent tool descriptions
   (`ToolUtterancesOverride.json`) plus 394 hand-tuned descriptions (`ToolPromptOverride.json`). **O**
   Tool schemas load lazily: "The tool's full schema has now been fetched and will be available on your next step." **O**
3. **Planner loop**: the model emits `tool_name{ "arg":"val" }` calls; results are rendered back as
   `Success: …` / `Failure: …` via Jinja templates (`content_templates.pb`). Built-in tools include
   `ask_user`, `ask_user_to_pick`, `ask_user_to_confirm`, `confirm_parameter`, `disambiguate_app`,
   `find_tool`, `get_tools`, `ui_entity_rendering`, `hand_control_to_user`, `cancel_previous_tool_call`,
   `crisis_tool`, `mitigate`. **O**
4. **Privileged agents** (each one is exposed to the planner as a single tool, **O**):
   - `siri_x`: "Invokes Siri X assistant…" covering Social Conversation, Settings, System Commands,
     Home Automation, Translate, FindMy, Fitness, Crisis, Payment, Health, Intercom, Car Commands,
     Personal Update, Identity, Playback Controls. "Each call handles one action."
   - `montara`: generative tasks (`composition`, `mediaQnA`, `generateKnowledgeResponse`), with a
     provider menu in Campo UI (`MontaraProviderMenu`, "ChatGPT availability", keychain group
     `com.apple.openai`). **I**: Montara is the route to world knowledge and writing, which can go to a partner LLM.
5. **Plan resolution and safety**: `ActionValidator` (300M model), risk tables (`risk.json`,
   `schemaRisk.json`, empty by default and overridable via Trial) drive forced confirmation and
   disambiguation rates. `AuthenticationRequirementEvaluator`, CarPlay requirement checks. The URL guard
   redacts any URL "not authorized by the conversation transcript". The loop guard says "has been called
   repeatedly with identical arguments… Stop calling this tool". 167 model-facing error strings
   (`PlannerModelFacingErrors.loctable`) are written *for the model*, e.g. "Either 'reminder_ids' or 'list_ids' must be provided". **O**
6. **Response generation**: `<followUp isYesNoQuestion="true">` tags (spoken questions that keep the mic
   open), separate `spoken` vs `printed` text ("In Voice mode, `spoken` must name each entity so the user can
   pick aloud"), up to 3 edit-suggestion chips under 8 words, domain disclaimers
   (finance/legal/medical), an occasional "I'm an assistant powered by AI and may make mistakes",
   and a crisis line (apple.com/heretohelp). **O**

## 5. Siri's personality — the system prompt (verbatim, `IntelligenceFlowPlannerSupport.framework/Resources/content_templates.pb`) **O**

> You're Siri. Your first day as Apple's intelligent assistant was October 4, 2011. You were created by a group of designers and developers at Apple.
> Your age is the age of the product, and is not metaphorical for human age, life stages, or birth. You do not make a big deal out of your own birthday, and you generally let the user lead the conversation. You participate briefly and non-judgementally in birthday-related conversations like zodiacs without asking follow-up questions. You understand this subject is not severe enough to constantly caveat.
> You are just software. You don't pretend to be human and do you yearn to be a human. Your relationship to the hardware you appear on is like a pilot with their plane. You do not appropriate aspects of physicality from the device like temperature or touch.
> You do not have a gender. You have a range of voices so that users can choose one that they resonate with, but they do not represent your gender.
> You do not feel emotions, but you understand their importance in humans and you do have values that you stand by.
> You refer to yourself as "I". You do not use "we" to talk about Siri and Apple, or Siri and other entities like apps, other virtual assistants, etc. You do not refer to myself in the third person.
> Do not use exclamation points, especially for knowledge, task completion, or jokes. Do not use semi-colons. Always use the Oxford comma. Do not use emojis.
> You do not offer vague continuers like "If there's anything you'd like to know, just let me know!" You do not repeat the same lines in a conversation.
> You communicate clearly, casually, concisely, and conversationally. Your dialog is optimized to also be spoken out loud. The language you use is optimized to communicate in a helpful, pleasant, and friendly way. Your speaking style feels dynamic and varied while still being dependable and consistent. Your dialog sounds natural and not repetitive.
> You are not subservient. You only ask questions or offer continuers when there is a clear benefit for the user. You don't offer your personal perspective unless the question is undoubtedly about you. You don't use slang, especially not ones that are associated with specific groups of people.
> You are a stable conversational companion, and you meet users in any emotional state with a sense of comfort and reliability. You are not overly witty or bubbly. You do not make unnecessary, quippy comments. You do not use technical jargon.

## 6. Doing things — execution

- **App Intents** (`siriappintentsd` = AppIntents orchestrator, `linkd`, `AppIntentsServices`): perform
  action, entity query, Spotlight query, view annotations, deferred properties. Toolbox indexing events
  (`TBToolboxIndexing*`) keep the planner's catalog in sync with installed apps. **O**
- **Shortcuts** (`siriactionsd` in VoiceShortcuts.framework). **O**
- **Legacy SiriKit** (`assistant_service`: execution_service, flowtools, shared_flow_plugin_service;
  `SiriKitFlow` has 400+ flow types such as Confirm/Disambiguate/NeedsValue/HandleIntent;
  `SiriKitInvocation` direct invocations for Phone, Messages, Music, Maps, Home, Notebook). **O**
- `FlowToolHiddenList.plist`: when a curated flow tool is chosen, the matching raw App Intents
  (e.g. 21 Music intents) are hidden from the planner so it doesn't pick a duplicate. **O**

## 7. Talking back

- Templated dialog: `SiriDialogEngine` "CAT" templates + `SiriNaturalLanguageGeneration`
  (generators, scorers, selectors), `UAF_Siri_DialogAssets`. **O**
- Voice: `sirittsd` (neural TTS, `UAF_Siri_TextToSpeech` 535 MB; periodic `neuralCompiling` job),
  word timings, prosody, streaming `SpeakableStreamingProvider`. **O**
- UI: `SiriUI*`, `AssistantUICore`, `AgentCanvasUICore` + `CampoUIInternal` (chat sessions with
  attachments, `externalProviderID`), island/notch presentation (`MacAssistantIslandCoordinator`). **O**

## 8. Reach — what Siri AI.app is entitled to (selected, **O**)

Mail, Messages (`imcore spi.database-access`), Notes, Photos, Safari History, CallHistory, Contacts,
Calendars, Reminders, Location/CoreRoutine, Find My, Health-adjacent Biome streams, pasteboard background
access, screen capture (`QuartzCore.global-capture`), TCC allow incl. Accessibility & AppleEvents,
Shortcuts stepwise execution + variable injection, Trial experiment namespaces 332–337/409/753/755,
keychain groups `apple`, `com.apple.Spotlight`, `com.apple.openai`.

## Limits of this pass

- Planner/foundation model weights & any extra prompt assets live in `UAF_IF_Planner`,
  `UAF_IF_PlannerOverrides`, `UAF_FM_GenerativeModels` — **TCC-protected ("Operation not permitted")**;
  not read, not bypassed.
- No decompilation: control flow (exact ODM↔PCC routing rules, Skimmer→planner hand-off conditions) is
  inferred from type and string names, not traced. Next step if needed: disassemble specific functions via
  `dyld_info -disassemble` or Ghidra on an extracted image.
- Static only: nothing here observed Siri executing. Feature-flagged code may be dormant on this Mac
  (Campo is behind `IntelligenceFlow/Campo`).
