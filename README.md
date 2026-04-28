# Aria — Background Mac Voice Agent
                                                                                                                                     
  A local-first voice assistant that runs silently on macOS. Press a hotkey, speak, and get a spoken answer — your screen never      
  moves.                                                                                                                             
                                                                                                                                     
  ## What it does         
  - **Voice-triggered** — hold ⌥ Space to record, release to process
  - **Background browser** — fetches and summarizes web results without stealing focus                                               
  - **Local transcription** — Whisper (faster-whisper) runs entirely on your machine  
  - **Wake word** — optional always-on trigger via OpenWakeWord                                                                      
  - **Job search** — searches LinkedIn and Indeed, caches results, tracks applications                                               
  - **Morning briefing** — weather, calendar, email, and news in one command                                                         
  - **Media control** — Apple Music and YouTube by voice                                                                             
  - **Mac system control** — open apps, adjust settings, take screenshots                                                            
  - **Multi-step tasks** — browser agent that plans and executes multi-step research                                                 
                                                                                                                                     
  ## Requirements                                                                                                                    
                                                                                                                                     
  - macOS (Apple Silicon or Intel)                                                                                                   
  - Python 3.11+                         
  - [Homebrew](https://brew.sh)          
  - `ffmpeg` — `brew install ffmpeg`     
                                         
  ## Setup                                                                                                                           
          
  ```bash                                                                                                                            
  # 1. Clone and enter the project
  git clone https://github.com/Bhanu1299/Aria.git
  cd Aria                                        
                                                                                                                                     
  # 2. Create virtualenv and install dependencies
  python3 -m venv venv                                                                                                               
  source venv/bin/activate                                                                                                           
  pip install -r requirements.txt
                                                                                                                                     
  # 3. Install Playwright's Chromium     
  playwright install chromium       
                             
  # 4. Copy and fill in your environment variables
  cp .env.example .env
                                                                                                                                     
  macOS Permissions
                                                                                                                                     
  Grant both of these before running, then restart your terminal:                                                                    
                          
  - Accessibility — System Settings → Privacy & Security → Accessibility → add your terminal app                                     
  - Microphone — System Settings → Privacy & Security → Microphone → add your terminal app
                                                                                                                                     
  Running                 
                                                                                                                                     
  source venv/bin/activate
  python main.py          
                          
  Aria starts silently with a menu bar icon. Hold ⌥ Space, speak your question, release.                                             
   
  ┌──────┬────────────────┐                                                                                                          
  │ Icon │     State      │
  ├──────┼────────────────┤
  │ ◉    │ Idle — waiting │
  ├──────┼────────────────┤              
  │ 🎙    │ Listening      │                                                                                                          
  ├──────┼────────────────┤
  │ ⏳   │ Thinking       │                                                                                                          
  ├──────┼────────────────┤
  │ ✓    │ Done           │
  └──────┴────────────────┘
                                         
  Stack                                                                                                                              
   
  ┌───────────────┬────────────────────────────────────┐                                                                             
  │     Layer     │                Tech                │
  ├───────────────┼────────────────────────────────────┤
  │ Transcription │ faster-whisper (base model, local) │
  ├───────────────┼────────────────────────────────────┤
  │ Browser       │ Playwright + Chromium (headless)   │                                                                             
  ├───────────────┼────────────────────────────────────┤                                                                             
  │ LLM           │ Groq (primary), Claude (fallback)  │                                                                             
  ├───────────────┼────────────────────────────────────┤                                                                             
  │ Hotkey        │ pynput                             │
  ├───────────────┼────────────────────────────────────┤                                                                             
  │ Audio         │ sounddevice / soundfile            │
  ├───────────────┼────────────────────────────────────┤                                                                             
  │ Menu bar      │ rumps                              │
  ├───────────────┼────────────────────────────────────┤
  │ Wake word     │ OpenWakeWord                       │
  └───────────────┴────────────────────────────────────┘

  Environment Variables                                                                                                              
   
  See .env.example for all options. Key ones:                                                                                        
                          
  GROQ_API_KEY=...        
  ANTHROPIC_API_KEY=...   
  BROWSER_TIMEOUT=30                     
                                                                                                                                     
  Troubleshooting
                                                                                                                                     
  See HOW_TO_RUN.md for common issues (hotkey not responding, microphone errors, auth problems).                                     
                          
  License                                                                                                                            
                                         
  MIT                     
