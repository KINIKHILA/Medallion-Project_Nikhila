import mimetypes
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import getaddresses
from pathlib import Path

from core.config import (
	SMTP_FROM,
	SMTP_HOST,
	SMTP_PASSWORD,
	SMTP_PORT,
	SMTP_TO,
	SMTP_USE_TLS,
	SMTP_USERNAME,
)


def send_report_email(report_path: str, business_intent: str, run_id: str) -> bool:
	"""Send the generated report as an email attachment when SMTP is configured.

	Return False when notifications are disabled, otherwise send the message and
	return True. Delivery errors are allowed to propagate to the caller.
	"""
	if not SMTP_HOST.strip() or not SMTP_TO.strip():
		return False

	sender = SMTP_FROM.strip() or SMTP_USERNAME.strip()
	if not sender:
		raise ValueError("Set SMTP_FROM or SMTP_USERNAME to send report notifications.")
	if bool(SMTP_USERNAME) != bool(SMTP_PASSWORD):
		raise ValueError("Set both SMTP_USERNAME and SMTP_PASSWORD, or leave both blank.")

	attachment_path = Path(report_path)
	if not attachment_path.is_file():
		raise FileNotFoundError(f"Report attachment not found: {attachment_path}")

	recipients = [address for _, address in getaddresses([SMTP_TO]) if address]
	if not recipients:
		raise ValueError("SMTP_TO must contain at least one valid email address.")

	message = EmailMessage()
	message["Subject"] = f"Pipeline report: {business_intent[:120]}"
	message["From"] = sender
	message["To"] = ", ".join(recipients)
	message.set_content(
		f"The Agentic Medallion Pipeline completed run {run_id}.\n\n"
		f"Business question: {business_intent}\n"
		"The generated report is attached."
	)
	content_type, _ = mimetypes.guess_type(attachment_path.name)
	maintype, subtype = (content_type or "application/octet-stream").split("/", 1)
	message.add_attachment(
		attachment_path.read_bytes(),
		maintype=maintype,
		subtype=subtype,
		filename=attachment_path.name,
	)

	with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
		if SMTP_USE_TLS:
			server.starttls(context=ssl.create_default_context())
		if SMTP_USERNAME:
			server.login(SMTP_USERNAME, SMTP_PASSWORD)
		server.send_message(message, to_addrs=recipients)
	return True