# Clinic Voice Agent 🎙️🩺

A small MVP voice assistant for **Sahyadri Family Clinic**, a fictional demo general-physician clinic. It answers clinic FAQs from a PDF knowledge base, provides demo clinic details, records local demo appointment requests, and routes uncertain or potentially urgent cases safely.

**Important:** The clinic, doctor (Dr. Michael Carter), phone number, address, hours, fees, and appointments are fictional demo data. Appointment tools do not connect to a real scheduling system. Medical content is not a clinical protocol and requires clinician approval before live use.

## Stack
- LiveKit Agents for real-time voice
- LangChain + Groq for tool-calling agent
- Docling for PDF parsing and chunking
- Sentence Transformers for embeddings
- Qdrant local vector store

## Setup
1. Python 3.12+
2. Install dependencies (this creates the `.venv` folder):
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
4. Configure LiveKit and Groq credentials:
   ```bash
   cp .env.example .env
   # edit .env
   ```
5. Ingest clinic documents (this recreates the local Qdrant collection):
   ```bash
   python ingest.py
   ```
6. Run:
   ```bash
   python -m voice_agent.main console
   ```

> Tip: if you don't want to activate the virtual environment, prefix the commands with `uv run`, for example `uv run python ingest.py` and `uv run python -m voice_agent.main console`.

## Example prompts
- "What are the clinic hours?"
- "How much is a consultation?"
- "I'd like to request an appointment tomorrow at 10 AM."
- "I need to reschedule my demo request."
- "Can I speak to clinic staff?"
- "I'm having trouble breathing." (tests emergency-safe routing; not diagnosis)

## Demo behavior and limits
- Appointment requests are stored in process memory only and reset when the process restarts.
- The tool says requests are not confirmed; there is no calendar or EHR integration.
- Handoff is logged locally only; it does not call a real clinic.
- Emergency number/transfer destination is intentionally not configured. Set and clinically review the local emergency script before any live use.
- Clinical education is disabled unless clinic-approved content is added.
- Do not use real patient information in this demo.

## Knowledge base
`docs/clinic-operations.pdf` is used by ingestion. `docs/clinic-operations.md` is the editable source. Replace the demo facts and get clinical/legal review before production.

## Operational note
`python ingest.py` recreates the configured Qdrant collection. Use a separate collection if you add other data, to avoid mixing documents.
