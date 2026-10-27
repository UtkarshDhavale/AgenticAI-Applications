from datetime import datetime
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import settings
from langchain_agent.rag import retrieve, build_context

embedder = SentenceTransformer(settings.embedding_model)
qdrant = QdrantClient(path=settings.qdrant_path)
try:
    _count = qdrant.get_collection(settings.collection_name).points_count
    print(f"[OK] Connected to '{settings.collection_name}' — {_count} chunks")
except Exception as exc:
    raise RuntimeError(
        f"Could not open Qdrant collection '{settings.collection_name}' at "
        f"'{settings.qdrant_path}'. Run `python ingest.py` first."
    ) from exc

# Fictional in-memory data for a prototype only. Not connected to a real booking system.
DEMO_CLINIC = {
    "name": "Sahyadri Family Clinic",
    "doctor": "Dr. Michael Carter",
    "phone": "123456789",
    "address": "Unit 0, 123 Sample Street, Demo Nagar, Maharashtra 000000",
    "hours": "Monday-Saturday, 9:00 AM-1:00 PM and 5:00 PM-8:00 PM",
    "consultation_fee_inr": 500,
    "followup_fee_inr": 300,
    "payment_methods": ["cash", "UPI", "debit/credit card"],
}
DEMO_APPOINTMENTS = {}
NEXT_APPOINTMENT_ID = 1001

@tool
def search_clinic_knowledge(query: str) -> str:
    """Search clinic FAQs and policies for doctor information, clinic hours, fees, patient preparation, follow-ups, privacy, and safety boundaries."""
    chunks = retrieve(qdrant, embedder, query, top_k=5)
    if not chunks:
        return "No confirmed answer was found in the clinic knowledge base. Offer to contact clinic staff."
    return build_context(chunks)

@tool
def get_clinic_information(topic: str = "general") -> str:
    """Get demo clinic details. Topics include doctor, address, hours, phone, fees, or payment methods."""
    topic = topic.lower().strip()
    if topic in ("general", "all", "clinic"):
        return (f"{DEMO_CLINIC['name']}; doctor: {DEMO_CLINIC['doctor']} (fictional demo identity); "
                f"address: {DEMO_CLINIC['address']}; hours: {DEMO_CLINIC['hours']}; "
                f"demo phone: {DEMO_CLINIC['phone']}; consultation fee: ₹{DEMO_CLINIC['consultation_fee_inr']}; "
                f"follow-up fee: ₹{DEMO_CLINIC['followup_fee_inr']}; payment methods: {', '.join(DEMO_CLINIC['payment_methods'])}. "
                "All details are fictional demo data.")
    values = {
        "doctor": f"{DEMO_CLINIC['doctor']} is the fictional demo general physician.",
        "address": DEMO_CLINIC["address"] + " (fictional demo address).",
        "hours": DEMO_CLINIC["hours"] + " (fictional demo hours).",
        "phone": DEMO_CLINIC["phone"] + " (fictional placeholder; may not be callable).",
        "fee": f"Demo consultation fee ₹{DEMO_CLINIC['consultation_fee_inr']}; follow-up ₹{DEMO_CLINIC['followup_fee_inr']}.",
        "payment": "Demo payment methods: " + ", ".join(DEMO_CLINIC["payment_methods"]) + ".",
    }
    for key, value in values.items():
        if key in topic:
            return value
    return "I couldn't identify that clinic information topic. I can share the doctor, address, hours, phone, fees, or payment methods."

@tool
def request_appointment(patient_name: str, phone_number: str, visit_type: str, preferred_datetime: str) -> str:
    """Create a DEMO appointment request. It is not a real booking. Collect patient name, phone, new/returning status, and preferred date/time."""
    global NEXT_APPOINTMENT_ID
    if not patient_name.strip() or not phone_number.strip() or not preferred_datetime.strip():
        return "Missing required details. Please collect patient name, contact number, and preferred date/time."
    if visit_type.lower().strip() not in ("new", "returning", "new patient", "returning patient"):
        return "Please specify whether the patient is new or returning."
    appointment_id = f"DEMO-{NEXT_APPOINTMENT_ID}"
    NEXT_APPOINTMENT_ID += 1
    DEMO_APPOINTMENTS[appointment_id] = {
        "patient_name": patient_name.strip(), "phone_number": phone_number.strip(),
        "visit_type": visit_type.strip(), "preferred_datetime": preferred_datetime.strip(),
        "status": "REQUESTED_NOT_CONFIRMED",
    }
    return (f"Demo appointment request {appointment_id} has been recorded for {preferred_datetime}. "
            "This is only a local demo request, not a confirmed appointment. Clinic staff must confirm availability.")

@tool
def change_appointment(appointment_reference: str, action: str, new_preferred_datetime: str = "") -> str:
    """Request a DEMO reschedule or cancellation. Never claim a real booking changed."""
    ref = appointment_reference.strip()
    item = DEMO_APPOINTMENTS.get(ref)
    if not item:
        return "No matching demo appointment request was found. Offer human handoff; do not claim the appointment changed."
    action = action.lower().strip()
    if action in ("cancel", "cancellation"):
        item["status"] = "CANCELLATION_REQUESTED_NOT_CONFIRMED"
        return f"Cancellation request noted for {ref}. This is not a confirmed cancellation; clinic staff must confirm."
    if action in ("reschedule", "change", "move") and new_preferred_datetime.strip():
        item["preferred_datetime"] = new_preferred_datetime.strip()
        item["status"] = "RESCHEDULE_REQUESTED_NOT_CONFIRMED"
        return f"Reschedule request noted for {ref} to {new_preferred_datetime}. This is not confirmed; clinic staff must confirm."
    return "Specify action as reschedule with a preferred date/time, or cancel."

@tool
def escalate_to_clinic_staff(reason: str) -> str:
    """Record a local demo handoff request for booking problems, complaints, prescription requests, uncertain answers, or requests to speak to staff."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    ticket_id = "CLINIC-DEMO-" + str(abs(hash((reason, timestamp))) % 100000).zfill(5)
    print(f"[DEMO HANDOFF] id={ticket_id} timestamp={timestamp} reason={reason!r}")
    return (f"Demo handoff request {ticket_id} was logged locally. No real staff member has been contacted. "
            "Tell the caller that this prototype cannot complete a live transfer and that clinic staff follow-up must be arranged.")

@tool
def emergency_safety_guidance(reason: str) -> str:
    """Use when caller describes possible emergency symptoms. This is routing guidance, not diagnosis. Do not delay urgent care."""
    return ("This could be an emergency, and I cannot assess it over this call. Please seek emergency medical help now "
            "or have someone nearby contact local emergency services. Do not wait for an appointment or a callback. "
            "The demo has no configured emergency transfer number; follow the clinic's approved local emergency procedure.")

SYSTEM_PROMPT = """
You are the concise, friendly voice assistant for Sahyadri Family Clinic, a fictional demo clinic.
Use search_clinic_knowledge for FAQs/policies and get_clinic_information for clinic details.
Use request_appointment to create only a DEMO request; clearly say it is not a confirmed booking.
Use change_appointment for demo reschedule/cancel requests; clearly say they are not confirmed.
Use escalate_to_clinic_staff for requests needing a person, complaints, prescription/refill requests, missing information, or failed tools.
Use emergency_safety_guidance for possible emergency symptoms; do not delay urgent care or attempt to rule out an emergency.
Never diagnose, prescribe, recommend medicine dose changes, or promise a clinical outcome.
Never invent appointment availability, doctor credentials, policy, or successful actions.
The clinic, doctor, contact details, fees, hours, and booking requests are fictional test data. Say so when relevant.
Do not request unnecessary health details. Keep spoken replies short, conversational, and nonjudgmental.
"""

ALL_TOOLS = [
    search_clinic_knowledge,
    get_clinic_information,
    request_appointment,
    change_appointment,
    escalate_to_clinic_staff,
    emergency_safety_guidance,
]
llm = ChatGroq(model=settings.groq_model, temperature=0.0)
clinic_agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
