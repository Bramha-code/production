import subprocess
import re
import time

print("Starting Cloudflare Tunnel...")
print("=" * 60)

# Start cloudflared
process = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

url_found = False
try:
    for line in process.stdout:
        print(line.strip())
        
        # Look for the URL
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if match and not url_found:
            url = match.group(0)
            print("\n" + "=" * 60)
            print(f"🎉 TUNNEL URL: {url}")
            print("=" * 60)
            print(f"\n📋 Chatbot: {url}/static/chat.html")
            print(f"📊 Dashboard: {url}/dashboard")
            print("=" * 60 + "\n")
            
            # Save to file
            with open("CURRENT_TUNNEL_URL.txt", "w") as f:
                f.write(f"Tunnel URL: {url}\n")
                f.write(f"Chatbot: {url}/static/chat.html\n")
                f.write(f"Dashboard: {url}/dashboard\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            url_found = True
        
        # Keep printing output
        if "Registered tunnel connection" in line:
            print("✅ Tunnel connection registered")
            
except KeyboardInterrupt:
    print("\n\nStopping tunnel...")
    process.terminate()
