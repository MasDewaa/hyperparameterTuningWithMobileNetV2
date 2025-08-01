# =========== WSL OPTIMIZATION GUIDE ===========

## 🚨 MASALAH DISK USAGE 100%

### Current Status:
- **Disk Usage:** 100%
- **VmmemWSL:** 733.6 MB/s disk activity
- **Memory:** 2,974.7 MB (VmmemWSL)

## 🔧 SOLUSI OPTIMASI:

### 1. **Buat WSL Config File:**
```bash
# Buat file konfigurasi WSL
sudo nano /etc/wsl.conf
```

**Isi dengan:**
```ini
[automount]
enabled = true
root = /mnt/
options = "metadata,umask=22,fmask=11"

[network]
generateHosts = false
generateResolvConf = false

[interop]
enabled = true
appendWindowsPath = false
```

### 2. **Optimasi Memory WSL:**
```bash
# Buat file .wslconfig di Windows
# Lokasi: C:\Users\[YourUsername]\.wslconfig
```

**Isi dengan:**
```ini
[wsl2]
memory=4GB
processors=4
swap=2GB
localhostForwarding=true
```

### 3. **Restart WSL:**
```bash
# Di Windows PowerShell (Admin):
wsl --shutdown
wsl
```

### 4. **Cleanup Temporary Files:**
```bash
# Bersihkan cache dan temp files
sudo apt clean
sudo apt autoremove
rm -rf /tmp/*
```

### 5. **Monitor Resource Usage:**
```bash
# Install htop untuk monitoring
sudo apt install htop
htop
```

## 📊 EXPECTED RESULTS:
- Disk usage: < 50%
- VmmemWSL memory: < 1GB
- Better performance untuk ML training

## ⚠️ WARNING:
- Restart WSL akan menutup semua terminal
- Backup work sebelum restart
- Monitor setelah restart 