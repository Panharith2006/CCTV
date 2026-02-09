"""
Script to get your Telegram Chat ID

INSTRUCTIONS:
1. Make sure your bot token is configured in .env file
2. Open Telegram app on your phone/computer
3. Search for your bot (the name you gave it when creating with @BotFather)
4. Send any message to your bot (example: /start or "Hello")
5. Run this script: python get_chat_id.py
6. Copy the chat_id shown and add it to .env file

"""
import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    print("⚠ python-dotenv not installed")
    print("  Install with: pip install python-dotenv")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

if not BOT_TOKEN:
    print("❌ ERROR: Bot token not found!")
    print("\nPlease add your bot token to .env file:")
    print("  TELEGRAM_BOT_TOKEN=your_token_here")
    sys.exit(1)

print("="*80)
print("TELEGRAM CHAT ID FINDER")
print("="*80)
print(f"\n✓ Bot Token: {BOT_TOKEN[:20]}..." if len(BOT_TOKEN) > 20 else f"\n✓ Bot Token: {BOT_TOKEN}")

# Try to get updates
try:
    import requests
    
    print("\n📱 Fetching recent messages from your bot...")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    response = requests.get(url)
    data = response.json()
    
    if not data.get("ok"):
        print(f"❌ Error from Telegram API: {data}")
        sys.exit(1)
    
    updates = data.get("result", [])
    
    if not updates:
        print("\n⚠ No messages found!")
        print("\n📋 INSTRUCTIONS:")
        print("  1. Open Telegram app")
        print("  2. Search for your bot")
        print("  3. Send ANY message to your bot (example: /start or Hello)")
        print("  4. Run this script again: python get_chat_id.py")
        sys.exit(0)
    
    print(f"\n✓ Found {len(updates)} message(s)")
    print("\n" + "="*80)
    print("YOUR CHAT IDs:")
    print("="*80)
    
    chat_ids = set()
    for update in updates:
        if "message" in update:
            chat_id = update["message"]["chat"]["id"]
            chat_type = update["message"]["chat"]["type"]
            first_name = update["message"]["chat"].get("first_name", "")
            username = update["message"]["chat"].get("username", "")
            
            chat_ids.add(chat_id)
            
            print(f"\n  Chat ID: {chat_id}")
            print(f"  Type: {chat_type}")
            if first_name:
                print(f"  Name: {first_name}")
            if username:
                print(f"  Username: @{username}")
    
    print("\n" + "="*80)
    
    if len(chat_ids) == 1:
        chat_id = list(chat_ids)[0]
        print(f"\n✅ YOUR CHAT ID: {chat_id}")
        print("\n📝 Copy this line to your .env file:")
        print(f"   TELEGRAM_CHAT_ID={chat_id}")
        
        # Try to update .env file automatically
        try:
            with open('.env', 'r') as f:
                content = f.read()
            
            if 'TELEGRAM_CHAT_ID=' in content:
                # Update existing
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith('TELEGRAM_CHAT_ID='):
                        new_lines.append(f'TELEGRAM_CHAT_ID={chat_id}')
                    else:
                        new_lines.append(line)
                
                with open('.env', 'w') as f:
                    f.write('\n'.join(new_lines))
                
                print("\n✅ .env file updated automatically!")
                print("\n🚀 You can now run: python main.py")
            else:
                # Add new
                with open('.env', 'a') as f:
                    f.write(f'\nTELEGRAM_CHAT_ID={chat_id}\n')
                
                print("\n✅ .env file updated automatically!")
                print("\n🚀 You can now run: python main.py")
        except Exception as e:
            print(f"\n⚠ Could not auto-update .env file: {e}")
            print("   Please copy the line manually")
    else:
        print(f"\n⚠ Found multiple chat IDs ({len(chat_ids)})")
        print("   Choose the one you want to use and add it to .env file")
    
    print("\n" + "="*80)
    
except ImportError:
    print("\n❌ 'requests' library not installed")
    print("   Install with: pip install requests")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
