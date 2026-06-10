"""
DigitDetox — Windows Laptop Bridge (Fixed)
"""
import subprocess, re, requests, time, psutil
from datetime import datetime, date

BACKEND_URL   = "http://localhost:5000/api/laptop-data"
INTERVAL_SECS = 30

SOCIAL_APPS = {
    "chrome","firefox","msedge","opera","brave",
    "discord","slack","telegram","whatsapp","zoom","teams",
}

# Expanded system skip list
SYSTEM_SKIP = {
    "system","system idle process","idle","registry","smss","csrss","wininit",
    "services","lsass","svchost","dwm","winlogon","fontdrvhost","sihost",
    "taskhostw","runtimebroker","startmenuexperiencehost","shellexperiencehost",
    "searchhost","searchapp","textinputhost","ctfmon","audiodg","conhost",
    "dllhost","msiexec","wuauclt","spoolsv","unsecapp","wmiprvse",
    "securityhealthservice","antimalware","backgroundtaskhost","useroobebroker",
    "lockapp","logonui","applicationframehost","systemsettings","explorer",
    # antivirus / security
    "msmpeng","mssense","securityhealthhost","sgrmbroker","smartscreen",
    "msedgewebview2","msedge webview2","webview2",
    # nvidia/amd/intel
    "nvdisplay","nvcontainer","amdow","igcc","igfxem","igfxtray",
    # other common system
    "wlanext","wificalling","phonexperiencehost","yourphone",
    "dllhost","taskmgr","cmd","powershell","python","pythonw",
    "node","git","ssh","sshd","putty",
}

NAME_MAP = {
    "msedge":     "Edge",
    "chrome":     "Chrome",
    "firefox":    "Firefox",
    "brave":      "Brave",
    "opera":      "Opera",
    "code":       "VS Code",
    "slack":      "Slack",
    "discord":    "Discord",
    "zoom":       "Zoom",
    "teams":      "Teams",
    "spotify":    "Spotify",
    "vlc":        "VLC",
    "winword":    "Word",
    "excel":      "Excel",
    "powerpnt":   "PowerPoint",
    "outlook":    "Outlook",
    "onenote":    "OneNote",
    "notepad":    "Notepad",
    "notepad++":  "Notepad++",
    "pycharm64":  "PyCharm",
    "idea64":     "IntelliJ",
    "devenv":     "Visual Studio",
    "obs64":      "OBS",
    "telegram":   "Telegram",
    "whatsapp":   "WhatsApp",
    "figma":      "Figma",
    "postman":    "Postman",
    "insomnia":   "Insomnia",
    "wpsoffice":  "WPS Office",
    "acrobat":    "Acrobat",
    "photoshop":  "Photoshop",
    "illustrator":"Illustrator",
}

def run_ps(script):
    try:
        r = subprocess.run(
            ["powershell","-NoProfile","-NonInteractive","-Command",script],
            capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except:
        return ""

def get_screen_time():
    today = date.today().strftime("%Y-%m-%d")
    now   = datetime.now()

    script = f"""
$start = '{today} 00:00:00'
$events = @()
try {{
    $e = Get-WinEvent -FilterHashtable @{{LogName='System';Id=1,42;StartTime=$start}} -ErrorAction SilentlyContinue
    if($e){{$events += $e}}
}} catch {{}}
try {{
    $e = Get-WinEvent -FilterHashtable @{{LogName='Security';Id=4800,4801;StartTime=$start}} -ErrorAction SilentlyContinue
    if($e){{$events += $e}}
}} catch {{}}
$events | Sort-Object TimeCreated | ForEach-Object {{
    "$($_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'))|$($_.Id)"
}}
"""
    out = run_ps(script)
    events = []
    for line in out.splitlines():
        line = line.strip()
        if "|" not in line: continue
        try:
            ts_str, eid = line.rsplit("|",1)
            ts  = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
            eid = int(eid.strip())
            if eid in (1, 4801):   events.append((ts,"ON"))
            elif eid in (42,4800): events.append((ts,"OFF"))
        except: continue

    # Fallback: use uptime
    if not events:
        try:
            boot = datetime.fromtimestamp(psutil.boot_time())
            today_start = datetime.strptime(today, "%Y-%m-%d")
            on_since = max(boot, today_start)
            mins = (now - on_since).total_seconds() / 60 * 0.85
            return round(mins/60, 2), 0.0, round(mins, 1)
        except:
            return 0.0, 0.0, 0.0

    midnight = datetime.strptime(f"{today} 00:00:00", "%Y-%m-%d %H:%M:%S")
    if events[0][1] == "OFF":
        events.insert(0, (midnight, "ON"))

    total = night = longest = 0.0
    i = 0
    while i < len(events):
        ts, et = events[i]
        if et == "ON":
            j = i+1
            while j < len(events) and events[j][1] != "OFF": j += 1
            t0 = ts
            t1 = events[j][0] if j < len(events) else now
            i  = j+1 if j < len(events) else i+1
            dur = (t1-t0).total_seconds()/60
            if 0 < dur < 720:
                total  += dur
                longest = max(longest, dur)
                if t0.hour >= 22 or t1.hour >= 22: night += dur
        else: i += 1

    return round(total/60,2), round(night/60,2), round(longest,1)


def get_app_usage():
    apps = {}
    cores = psutil.cpu_count() or 4
    try:
        for proc in psutil.process_iter(['name','cpu_times','status']):
            try:
                raw_name = proc.info['name'] or ""
                name = raw_name.lower().replace('.exe','').strip()

                # Skip system processes
                if name in SYSTEM_SKIP: continue
                if not name or len(name) < 2: continue
                # Skip names that are purely numbers or single letters
                if name.isdigit(): continue

                cpu = proc.info['cpu_times']
                if not cpu: continue
                cpu_secs = cpu.user + cpu.system
                if cpu_secs < 2: continue  # skip processes with < 2s CPU time

                display = NAME_MAP.get(name, name.capitalize())
                apps[display] = apps.get(display, 0) + cpu_secs
            except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    except: pass

    # Convert to minutes, normalize by cores
    app_mins = {}
    for name, secs in apps.items():
        mins = round(secs / cores / 60, 1)
        if mins >= 0.1:
            app_mins[name] = mins

    sorted_apps = sorted(app_mins.items(), key=lambda x: x[1], reverse=True)
    top5 = [{"app": n, "mins": m} for n,m in sorted_apps[:5]]
    total_opens = len(app_mins)
    social_mins = sum(m for n,m in app_mins.items()
                      if any(s in n.lower() for s in SOCIAL_APPS))
    return top5, total_opens, round(social_mins/60, 2)


def get_notifications():
    today = date.today().strftime("%Y-%m-%d")
    # Try Windows notification platform log
    script = f"""
try {{
    $n = (Get-WinEvent -FilterHashtable @{{
        LogName='Microsoft-Windows-PushNotifications-Platform/Operational';
        StartTime='{today} 00:00:00'
    }} -ErrorAction SilentlyContinue).Count
    if($n){{$n}}else{{0}}
}} catch {{ 0 }}
"""
    out = run_ps(script)
    try:
        n = int(out.strip())
        if n > 0: return n
    except: pass

    # Fallback: count toast notifications from app event log
    script2 = f"""
try {{
    $n = (Get-WinEvent -FilterHashtable @{{
        LogName='Microsoft-Windows-Notifications/Operational';
        StartTime='{today} 00:00:00'
    }} -ErrorAction SilentlyContinue).Count
    if($n){{$n}}else{{0}}
}} catch {{ 0 }}
"""
    out2 = run_ps(script2)
    try:
        n2 = int(out2.strip())
        if n2 > 0: return n2
    except: pass

    return 0


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║      DigitDetox — Windows Laptop Bridge             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  📡 Backend: {BACKEND_URL}")
    print(f"  🔄 Every {INTERVAL_SECS}s  (Ctrl+C to stop)\n")

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now()
        print(f"[Cycle #{cycle} — {now.strftime('%H:%M:%S')}]")
        print("=" * 52)

        screen_hrs, night_hrs, longest = get_screen_time()
        top5, app_opens, social_hrs    = get_app_usage()
        notifications                  = get_notifications()

        h = int(screen_hrs)
        m = int((screen_hrs % 1) * 60)

        print(f"  💻 Screen Hrs    : {screen_hrs:.2f}  ({h}h {m:02d}m)")
        print(f"  🔔 Notifications : {notifications}")
        print(f"  🌙 Night Hrs     : {night_hrs:.2f}")
        print(f"  📲 Active Apps   : {app_opens}")
        print(f"  📱 Social Hrs    : {social_hrs:.2f}")
        print(f"  ⏱  Longest Sess  : {longest} mins")
        if top5:
            print("  🏆 Top Apps:")
            max_m = top5[0]['mins'] or 1
            for i, a in enumerate(top5, 1):
                bar = "█" * max(1, int(a['mins']/max_m*15))
                print(f"     {i}. {a['app']:<22} {a['mins']:>5} min  {bar}")
        print("=" * 52)

        payload = {
            "screen_time":     screen_hrs,
            "notifications":   notifications,
            "night_usage":     night_hrs,
            "app_opens":       app_opens,
            "social_media":    social_hrs,
            "longest_session": longest,
            "top_apps":        top5,
            "source":          "live",
        }
        try:
            r = requests.post(BACKEND_URL, json=payload, timeout=5)
            print(f"  ✅ Sent to backend ({r.status_code})")
        except:
            print("  ❌ Backend offline — run: python app.py")

        print(f"  Next sync in {INTERVAL_SECS}s…\n")
        time.sleep(INTERVAL_SECS)

if __name__ == "__main__":
    main()
