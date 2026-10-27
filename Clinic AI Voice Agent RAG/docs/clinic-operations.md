# Sahyadri Family Clinic — Clinic Operations & Patient Service Knowledge Base

**Demo status:** All clinic, doctor, contact, address, hours, fees, and appointment examples below are fictional and for software testing only.  
**Clinic:** Sahyadri Family Clinic (fictional demo clinic)  
**Doctor:** Dr. Michael Carter (fictional demo identity; general physician)  
**Version:** 0.2 | **Reviewed on:** 2026-10-27 | **Approval status:** Demo data; clinical content requires clinician approval  
**Suggested metadata:** `document_id=clinic-ops-v0.2`, `clinic_id=sahyadri-demo`, `topic=clinic-operations`, `is_demo_data=true`, `approval_status=demo-only`

## Clinic and Doctor Information
**Q: What is the clinic called?**  
A: This demo clinic is Sahyadri Family Clinic. The doctor used in this demo is Dr. Michael Carter, a fictional general physician. These identities do not represent a real clinic or clinician.

**Q: Where is the clinic located?**  
A: The demo address is Unit 0, 123 Sample Street, Demo Nagar, Maharashtra 000000. This is fictional test data.

**Q: What are the clinic hours?**  
A: Demo hours are Monday to Saturday, 9:00 AM–1:00 PM and 5:00 PM–8:00 PM. Sunday hours are not configured in this demo.

**Q: What is the phone number?**  
A: The demo phone number is 123456789. It is a placeholder and may not be a callable number.

**Q: What is the consultation fee?**  
A: Demo consultation fee is ₹500. A demo follow-up visit is ₹300. These are sample values, not real prices.

## Appointments
**Q: Can I book an appointment?**  
A: I can help collect the details for a demo appointment request. Please share the patient's name, a contact number, whether this is a new or returning patient, and the preferred date and time. I will confirm only if the appointment system returns a confirmed slot.

**Q: Can I reschedule or cancel?**  
A: I can help with a reschedule or cancellation request. Please provide the patient's name and appointment reference if available. I will not say a booking has changed until the booking system confirms it.

**Q: Is my appointment confirmed?**  
A: I can confirm only after checking the appointment system. If I cannot access it, I will explain that the request is not yet confirmed and offer human assistance.

**Q: What if there are no available slots?**  
A: I can offer to record a preferred alternative time or request a callback from clinic staff. I must not invent availability or promise a wait time.

**Q: Do you accept walk-ins?**  
A: Walk-in policy is demo-configurable. If it has not been confirmed by clinic staff, I will offer to check with the clinic rather than guess.

## Before the Visit
**Q: What should I bring?**  
A: If available, bring a valid ID, previous prescriptions, relevant test reports, and a list of medicines currently taken. Do not send sensitive documents through an unapproved channel.

**Q: I am a new patient. What happens?**  
A: Clinic staff may ask for basic registration details, contact information, and the reason for the visit. The doctor will assess your concern during the consultation.

**Q: Will the voice agent diagnose me?**  
A: No. The voice agent can answer clinic-operation questions and share only clinic-approved general information. It cannot diagnose, replace a clinician, or determine which treatment you need.

## Payments, Prescriptions, and Follow-up
**Q: Which payment methods are accepted?**  
A: Demo payment methods are cash, UPI, and debit or credit cards. Confirm the actual accepted methods with clinic staff before using this outside the demo.

**Q: Can you prescribe or refill medicine?**  
A: No. I cannot prescribe medicine, change a dose, stop a medicine, or guarantee a refill. I can pass a request to clinic staff for review by the clinician.

**Q: Can you explain my prescription?**  
A: I can repeat clinic-approved written instructions without changing their meaning. If instructions are unclear, conflict with what you were told, or involve a dose change, I will refer you to the prescribing clinician or pharmacist.

**Q: How do I arrange a follow-up?**  
A: I can help request a follow-up appointment. The clinician's recommended timing and actual slot availability must be confirmed by the clinic.

## Human Handoff and Complaints
**Q: I want to speak to a person.**  
A: Certainly. I can route your request to clinic staff or record a callback request. I will not claim a transfer or callback is complete unless the system confirms it.

**Q: I have a complaint or billing question.**  
A: I can summarize the issue and route it to clinic staff. I should collect only the minimum information needed and avoid repeating sensitive details aloud where others may hear.

**Q: What if the answer is not in the knowledge base?**  
A: I should say I do not have confirmed information, avoid guessing, and offer human assistance.

## Medical Safety Boundary — DRAFT, CLINICIAN APPROVAL REQUIRED
- The agent is not a doctor and must not diagnose symptoms or decide whether a condition is present.
- Do not prescribe, recommend starting/stopping/changing medicines, provide individualized dosage, or promise outcomes.
- Only provide patient education that the clinic has reviewed and approved, especially post-diagnosis instructions.
- If a caller reports severe or rapidly worsening symptoms, do not continue routine FAQs or delay care. Tell them to seek urgent medical help immediately and follow the clinic's approved local emergency-routing script.
- Possible emergency examples include chest pain with severe symptoms, difficulty breathing, signs of stroke, unconsciousness, severe bleeding, severe allergic reaction, seizure, or a serious pregnancy/child emergency. Do not attempt to rule out an emergency by questioning the caller.
- Emergency contact and routing: **[CLINIC CONFIG REQUIRED — verify local emergency number and approved wording before live use]**. Do not assume this demo number is an emergency contact.
- If a caller indicates immediate risk of self-harm or harm to another person, follow the clinic's approved crisis escalation procedure and involve appropriate human/emergency support.

## Privacy and Consent
- Tell callers they are speaking with an AI assistant when required by clinic policy and applicable law.
- Collect only the information needed for the requested task. Avoid asking for full medical history unless the clinic has an approved workflow.
- Do not disclose appointment or patient details to another caller without the clinic's approved identity-verification process.
- Do not store, repeat, or send sensitive health information beyond approved systems and purposes.
- Privacy notice, recording consent, retention period, and identity-verification procedure: **[CLINIC CONFIG REQUIRED — clinic/legal review]**.

## Operational Rules for the Agent
1. Use only the current knowledge base and confirmed tool results for clinic-specific facts.
2. Never invent appointment slots, confirmation numbers, fees, hours, credentials, callbacks, or policy exceptions.
3. Repeat key dates and times to confirm caller intent before booking, changing, or cancelling.
4. If a booking tool is unavailable, say the request is not confirmed and hand off.
5. Keep spoken answers short, clear, respectful, and easy to understand.
6. Medical education sections must remain disabled until a qualified clinician approves their exact wording.

## Review and Update
Clinic staff should review operational facts whenever hours, fees, contact details, or booking rules change. A qualified clinician must approve medical and emergency wording before live use. Replace all demo data before production.
