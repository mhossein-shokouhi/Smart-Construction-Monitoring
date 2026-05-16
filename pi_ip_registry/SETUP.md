# Pi IP registry (Google Sheet)

When university Wi‑Fi blocks mDNS, each Raspberry Pi reports its **wlan0** (or fallback) IPv4 address to a **Google Sheet** every minute. You open the sheet (or run a small fetch script on your laptop) to see Pi IDs **0–3** and their current IPs.

No changes to existing project files — everything lives under `pi_ip_registry/`.

---

## 1. Create the Google Sheet

1. Go to [Google Sheets](https://sheets.google.com) → **Blank spreadsheet**.
2. Name it e.g. `Rogers Pi IP Registry`.
3. **Extensions → Apps Script**.
4. Delete any default code and paste the contents of `google_apps_script/Code.gs`.
5. **Project settings** (gear) → **Script properties** → Add:
   - Name: `REGISTRY_SECRET`
   - Value: a long random string (save this — Pis and your laptop need it).
6. **Deploy → New deployment**:
   - Type: **Web app**
   - Execute as: **Me**
   - Who has access: **Anyone** (the Pis only know your shared secret; they are not Google users)
7. Copy the **Web app URL** (ends in `/exec`). That is `REGISTRY_URL`.

Optional: run `setupSheet` once from the editor (select `getOrCreateSheet_` won’t work as a runner — just deploy; the first Pi POST creates the **Pi Registry** tab).

---

## 2. On each Raspberry Pi (IDs 0, 1, 2, 3)

```bash
# From your repo on the Pi
cd ~/Final_Demo/pi_ip_registry

sudo cp config.example.env /etc/pi-ip-announce.env
sudo nano /etc/pi-ip-announce.env
```

Set on **each** Pi:

| Variable | Pi 0 | Pi 1 | Pi 2 | Pi 3 |
|----------|------|------|------|------|
| `PI_ID` | 0 | 1 | 2 | 3 |
| `REGISTRY_URL` | same web app URL on all | | | |
| `REGISTRY_SECRET` | same secret on all | | | |

Test once:

```bash
export $(grep -v '^#' /etc/pi-ip-announce.env | xargs)
python3 announce_ip.py
# Ctrl+C after you see "Announced pi_id=..."
```

### Run on boot (systemd)

Edit `systemd/pi-ip-announce.service` if your repo path is not `/home/pi/Final_Demo`, then:

```bash
sudo cp systemd/pi-ip-announce.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pi-ip-announce.service
sudo systemctl status pi-ip-announce.service
```

---

## 3. On your laptop

**Option A — open the sheet** in the browser. The **Pi Registry** tab updates `last_seen_utc` every heartbeat.

**Option B — CLI:**

```bash
cd pi_ip_registry
export REGISTRY_URL='https://script.google.com/macros/s/.../exec'
export REGISTRY_SECRET='your-secret'
python3 fetch_registry.py
```

Then update `cameras.json` `pi_host` fields manually from the IPs shown (this repo’s registry script does not edit `cameras.json`).

---

## How it works

```
  Pi 0..3 (boot)                    Google                    You
  announce_ip.py  ──POST JSON──▶  Apps Script Web App  ──▶  Sheet row
  every 60s         secret+pi_id+ip     doPost()              pi_id, ip, last_seen
```

- **POST** body: `{ "secret", "pi_id", "ip", "interface", "hostname" }`
- **GET** `?secret=...` returns all rows (used by `fetch_registry.py`)
- If DHCP changes the IP, the next heartbeat overwrites the row for that `pi_id`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `unauthorized` | `REGISTRY_SECRET` must match Script property and Pi env |
| `No IP on network interfaces` | Wi‑Fi not up yet; service retries. Check `NETWORK_INTERFACE` |
| POST works from browser but not Pi | Use the `/exec` URL from deployment, not `/dev` |
| Sheet empty | Run one Pi with `announce_ip.py` and check `journalctl -u pi-ip-announce` |

---

## Security note

The web app is public URL + shared secret (like an API key). Use a strong `REGISTRY_SECRET` and do not commit `/etc/pi-ip-announce.env` to git.
