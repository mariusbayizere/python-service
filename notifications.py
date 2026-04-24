"""
notifications.py — DriveGuardAI Notification Service
Handles SMS (Twilio) and Email (via Java backend).
"""
import threading
from datetime import datetime

import requests

import config

# ── Twilio setup ──────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("twilio not available — pip install twilio")


def send_sms_alert(alert_name: str, current_session: dict) -> None:
    """Send SMS for CRITICAL violations only via Twilio."""
    if not TWILIO_AVAILABLE:
        print("SMS: Twilio library not installed")
        return
    if config.TWILIO_ACCOUNT_SID == "YOUR_TWILIO_ACCOUNT_SID":
        print("SMS: Twilio not configured in .env")
        return
    if alert_name not in config.CRITICAL_VIOLATIONS:
        return

    driver_name  = current_session.get('driver_name')  or "Driver #" + str(current_session.get('driver_id'))
    vehicle_name = current_session.get('vehicle_name') or "Vehicle #" + str(current_session.get('vehicle_id', '?'))
    trip_name    = current_session.get('trip_name')    or "Trip #"    + str(current_session.get('trip_id'))
    label        = config.VIOLATION_LABELS.get(alert_name, alert_name.replace('_', ' ').upper())
    now          = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    body = (
        f"DriveGuardAI CRITICAL ALERT\n\n"
        f"Driver {driver_name} was detected\n"
        f"doing {label}\n"
        f"while on {trip_name} in {vehicle_name}.\n\n"
        f"Time: {now}\n"
        f"Please take immediate action."
    )

    def _run():
        try:
            client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
            msg    = client.messages.create(
                body=body,
                from_=config.TWILIO_FROM_NUMBER,
                to=config.MANAGER_PHONE,
            )
            print(f"SMS sent — SID: {msg.sid} | Alert: {alert_name}")
        except Exception as e:
            print(f"SMS error: {e}")

    threading.Thread(target=_run, daemon=True).start()


def trigger_email(violation_data: dict, current_session: dict) -> None:
    """Send violation email via the Java backend."""
    def _run():
        try:
            driver_name  = current_session.get('driver_name')  or "Driver #"  + str(violation_data['driver_id'])
            vehicle_name = current_session.get('vehicle_name') or "Vehicle #" + str(violation_data['vehicle_id'])
            trip_name    = current_session.get('trip_name')    or "Trip #"    + str(violation_data['trip_id'])

            body = (
                f"A violation was detected by the AI monitoring system.\n\n"
                f"Driver      : {driver_name}\n"
                f"Vehicle     : {vehicle_name}\n"
                f"Trip        : {trip_name}\n"
                f"Violation   : {violation_data['incident_type']}\n"
                f"Severity    : {violation_data['severity']}\n"
                f"Description : {violation_data['description']}\n"
                f"Time        : {violation_data['timestamp']}\n"
                f"Screenshot  : {violation_data.get('screenshot', 'N/A')}\n\n"
                f"Please log in to the DriveGuardAI dashboard to review."
            )
            subject = (
                f"[DriveGuardAI] {violation_data['incident_type']} — "
                f"{driver_name} ({violation_data['severity']})"
            )
            payload = {
                "recipient": "bayizeremarius119@gmail.com",
                "subject":   subject,
                "msgBody":   body,
            }
            r = requests.post(
                config.JAVA_BACKEND_URL + "/api/v1/notifications/email",
                json=payload,
                timeout=10,
            )
            print(f"Email triggered — status {r.status_code}")
        except Exception as e:
            print(f"Email error: {e}")

    threading.Thread(target=_run, daemon=True).start()
