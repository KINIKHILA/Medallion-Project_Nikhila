from email.message import EmailMessage
from pathlib import Path

from core import email_notifications


class FakeSMTP:
	def __init__(self):
		self.started_tls = False
		self.login_args = None
		self.sent_message = None
		self.recipients = None

	def __enter__(self):
		return self

	def __exit__(self, *args):
		return False

	def starttls(self, context=None):
		self.started_tls = True

	def login(self, username, password):
		self.login_args = (username, password)

	def send_message(self, message, to_addrs=None):
		self.sent_message = message
		self.recipients = to_addrs


def test_email_is_skipped_without_smtp_host(monkeypatch, tmp_path):
	monkeypatch.setattr(email_notifications, "SMTP_HOST", "")
	monkeypatch.setattr(email_notifications, "SMTP_TO", "person@example.com")

	assert email_notifications.send_report_email(
		str(tmp_path / "missing.html"), "Sales summary", "test-run"
	) is False


def test_report_is_sent_as_html_attachment(monkeypatch, tmp_path):
	report_path = tmp_path / "report.html"
	report_path.write_text("<html><body>Report</body></html>", encoding="utf-8")
	smtp = FakeSMTP()
	monkeypatch.setattr(email_notifications, "SMTP_HOST", "smtp.example.com")
	monkeypatch.setattr(email_notifications, "SMTP_PORT", 587)
	monkeypatch.setattr(email_notifications, "SMTP_USERNAME", "sender@example.com")
	monkeypatch.setattr(email_notifications, "SMTP_PASSWORD", "test-only-password")
	monkeypatch.setattr(email_notifications, "SMTP_FROM", "reports@example.com")
	monkeypatch.setattr(
		email_notifications, "SMTP_TO", "recipient@example.com, second@example.com"
	)
	monkeypatch.setattr(email_notifications, "SMTP_USE_TLS", True)
	monkeypatch.setattr(email_notifications.smtplib, "SMTP", lambda *args, **kwargs: smtp)

	assert email_notifications.send_report_email(
		str(report_path), "Sales summary", "test-run"
	) is True

	assert smtp.started_tls is True
	assert smtp.login_args == ("sender@example.com", "test-only-password")
	assert smtp.recipients == ["recipient@example.com", "second@example.com"]
	assert isinstance(smtp.sent_message, EmailMessage)
	attachments = list(smtp.sent_message.iter_attachments())
	assert len(attachments) == 1
	assert attachments[0].get_filename() == "report.html"
	assert attachments[0].get_content_type() == "text/html"
	assert "Report" in attachments[0].get_content()