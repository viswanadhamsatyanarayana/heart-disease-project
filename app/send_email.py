import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Step 1: Email and App Password
sender_email = "ranjithpaul40@gmail.com"       # Replace with your Gmail
app_password = "fyds zdjp eovt gznu"        # Your generated App Password
receiver_email = "viswanadhamsatyanarayana17@gmail.com" # Replace with recipient email

# Step 2: Create the email
subject = "Test Email from Python"
body = "Hello, this is a test email sent using Python and Gmail App Password."

msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = subject
msg.attach(MIMEText(body, "plain"))

# Step 3: Connect to Gmail server and send
try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()
    print("✅ Email sent successfully!")
except Exception as e:
    print("❌ Error:", e)
