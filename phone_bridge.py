"""
DigitDetox — Phone Bridge (USB/ADB) - Fixed app opens
"""
import subprocess, re, requests, time
from datetime import datetime

BACKEND_URL   = "http://localhost:5000/api/phone-data"
INTERVAL_SECS = 60

SOCIAL_APPS = {
    "instagram","whatsapp","snapchat","facebook","twitter","tiktok",
    "telegram","messenger","discord","linkedin","reddit","youtube",
    "hinge","bumble","tinder","schmooze",
}
SYSTEM_SKIP = {
    "android","systemui","launcher3","launcher","permissioncontroller",
    "packageinstaller","settings","inputmethod","keyguard","wallpaper",
    "deskclock","dialer","phone","contacts","calendar","clock",
}

def adb(cmd):
    try:
        r = subprocess.run(
            ["adb","shell"] + cmd.split(),
            capture_output=True, timeout=30
        )
        return r.stdout.decode("utf-8", errors="ignore")
    except:
        return ""

def short_name(pkg):
    parts = pkg.split(".")
    for c in reversed(parts):
        cl = c.lower()
        if cl not in ("com","org","net","android","app","apps","client","google"):
            return cl
    return parts[-1].lower()

def is_system(pkg):
    n = short_name(pkg)
    return (n in SYSTEM_SKIP or
            any(s in pkg.lower() for s in
                ("android.server","google.gms","google.gsf","qualcomm","mediatek","bbk","vivo")))

def get_screen_time(today):
    raw = adb("dumpsys usagestats --table")

    # Collect only unique timestamps — deduplicate by timestamp only (ignore duplicates across sections)
    on_times  = set()
    off_times = set()

    for line in raw.splitlines():
        if today not in line: continue
        if "SCREEN_INTERACTIVE" in line or "SCREEN_NON_INTERACTIVE" in line:
            m = re.search(r'time="(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"', line)
            if m:
                ts = m.group(1)
                if "SCREEN_NON_INTERACTIVE" in line:
                    off_times.add(ts)
                else:
                    on_times.add(ts)

    # Build clean event list — if a timestamp is in both, treat as OFF (screen turned off)
    events = []
    for ts in on_times - off_times:
        events.append((ts, "ON"))
    for ts in off_times:
        events.append((ts, "OFF"))
    events.sort(key=lambda x: x[0])

    # If no events, return 0
    if not events:
        return 0.0

    total = 0.0
    i = 0
    while i < len(events):
        ts, et = events[i]
        if et == "ON":
            j = i+1
            while j < len(events) and events[j][1] != "OFF": j += 1
            t0 = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if j < len(events):
                t1 = datetime.strptime(events[j][0], "%Y-%m-%d %H:%M:%S")
                i = j+1
            else:
                t1 = datetime.now()
                i += 1
            dur = (t1-t0).total_seconds()/60
            # Only count sessions between 10 seconds and 2 hours
            if 0.17 < dur < 120:
                total += dur
        else:
            i += 1
    return round(total/60, 2)

def get_unlocks(today):
    raw = adb("dumpsys usagestats --table")
    seen = set()
    for line in raw.splitlines():
        if today in line and "KEYGUARD_HIDDEN" in line:
            m = re.search(r'time="([^"]+)"', line)
            if m: seen.add(m.group(1))
    return len(seen)

def get_notifications(today):
    raw = adb("dumpsys usagestats --table")
    seen = set()
    for line in raw.splitlines():
        if today in line and "NOTIFICATION_INTERRUPTION" in line:
            m = re.search(r'time="([^"]+)".*?package=(\S+)', line)
            if m: seen.add((m.group(1), m.group(2)))
    return len(seen)

def get_night_usage(today):
    raw = adb("dumpsys usagestats --table")
    events, seen = [], set()
    for line in raw.splitlines():
        if today not in line: continue
        if "SCREEN_INTERACTIVE" in line or "SCREEN_NON_INTERACTIVE" in line:
            m = re.search(r'time="(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"', line)
            if m:
                ts = m.group(1)
                et = "ON" if "SCREEN_INTERACTIVE" in line else "OFF"
                key = (ts, et)
                if key not in seen:
                    seen.add(key); events.append((ts, et))
    events.sort(key=lambda x: x[0])
    night = 0.0
    i = 0
    while i < len(events):
        ts, et = events[i]
        if et == "ON":
            j = i+1
            while j < len(events) and events[j][1] != "OFF": j += 1
            t0 = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(events[j][0], "%Y-%m-%d %H:%M:%S") if j < len(events) else datetime.now()
            i = j+1 if j < len(events) else i+1
            dur = (t1-t0).total_seconds()/60
            if 0 < dur < 180 and (t0.hour >= 22 or t1.hour >= 22): night += dur
        else: i += 1
    return round(night/60, 2)

def get_apps(today):
    raw = adb("dumpsys usagestats")

    # Collect ALL sections that contain today's date
    # Try multiple split strategies to find the daily bucket
    all_apps = {}

    # Strategy: find blocks between "android.content.pm" style headers
    # and parse all appLaunchCount lines that appear near today's date
    lines = raw.splitlines()
    current_pkg = None
    current_has_today = False
    current_launches = 0

    for line in lines:
        # New package block
        pm = re.search(r'package=([^\s]+)', line)
        if pm:
            # Save previous
            if current_pkg and current_has_today and current_launches > 0:
                if not is_system(current_pkg):
                    name = short_name(current_pkg)
                    existing = all_apps.get(name, 0)
                    # Keep the SMALLER value (daily, not all-time)
                    if existing == 0 or current_launches < existing:
                        all_apps[name] = current_launches
            # Start new
            current_pkg = pm.group(1)
            current_has_today = False
            current_launches = 0

        if today in line:
            current_has_today = True

        lm = re.search(r'appLaunchCount=(\d+)', line)
        if lm:
            val = int(lm.group(1))
            # Only update if this is smaller (= more recent/daily bucket)
            if val > 0:
                if current_launches == 0 or val < current_launches:
                    current_launches = val

    # Save last block
    if current_pkg and current_has_today and current_launches > 0:
        if not is_system(current_pkg):
            name = short_name(current_pkg)
            existing = all_apps.get(name, 0)
            if existing == 0 or current_launches < existing:
                all_apps[name] = current_launches

    # Filter out unrealistic values (> 500 = all-time, not daily)
    all_apps = {k: v for k, v in all_apps.items() if 0 < v <= 500}

    if not all_apps:
        return [], 0, 0.0

    sorted_apps = sorted(all_apps.items(), key=lambda x: x[1], reverse=True)
    top5 = [{"app": a.capitalize(), "opens": o} for a,o in sorted_apps[:5]]
    total_opens = sum(all_apps.values())
    social_hrs = round(sum(o for a,o in all_apps.items() if a in SOCIAL_APPS)*2/60, 1)
    return top5, total_opens, social_hrs

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║        DigitDetox — Phone Bridge (USB/ADB)          ║")
    print("╚══════════════════════════════════════════════════════╝")
    devices = subprocess.run(["adb","devices"], capture_output=True, text=True).stdout
    if "\tdevice" not in devices:
        print("  ❌  No phone found!")
        print("  → Connect USB cable")
        print("  → Enable USB Debugging: Settings → Developer Options → USB Debugging ON")
        return
    print("  ✅  Phone connected!")
    print(f"  🔄 Syncing every {INTERVAL_SECS}s (Ctrl+C to stop)\n")

    cycle = 0
    while True:
        cycle += 1
        now   = datetime.now()
        today = now.strftime("%Y-%m-%d")
        print(f"[Cycle #{cycle} — {now.strftime('%H:%M:%S')}]")
        print("="*50)

        screen_hrs    = get_screen_time(today)
        unlocks       = get_unlocks(today)
        notifications = get_notifications(today)
        night_hrs     = get_night_usage(today)
        top5, opens, social_hrs = get_apps(today)

        h, m = int(screen_hrs), int((screen_hrs%1)*60)
        print(f"  📱 Screen Hrs    : {screen_hrs:.2f} ({h}h {m:02d}m)")
        print(f"  🔓 Unlocks       : {unlocks}")
        print(f"  🔔 Notifications : {notifications}")
        print(f"  🌙 Night Hrs     : {night_hrs:.2f}")
        print(f"  📲 App Opens     : {opens}")
        print(f"  📱 Social Hrs    : {social_hrs:.2f}")
        if top5:
            print("  🏆 Top Apps:")
            max_o = top5[0]['opens'] or 1
            for i,e in enumerate(top5,1):
                bar = "█" * max(1, int(e['opens']/max_o*15))
                print(f"     {i}. {e['app']:<20} {e['opens']:>3} opens  {bar}")
        print("="*50)

        payload = {
            "screen_time": screen_hrs, "screen_unlocks": unlocks,
            "notifications": notifications, "night_usage": night_hrs,
            "app_opens": opens, "social_media": social_hrs,
            "longest_session": 30, "top_apps": top5, "source": "live",
        }
        try:
            r = requests.post(BACKEND_URL, json=payload, timeout=5)
            print(f"  ✅ Sent to backend ({r.status_code})")
        except:
            print("  ❌ Backend offline — run python app.py first")
        print(f"  Next sync in {INTERVAL_SECS}s…\n")
        time.sleep(INTERVAL_SECS)

if __name__ == "__main__":
    main()