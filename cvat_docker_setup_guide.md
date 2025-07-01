# CVAT Docker Kurulum Rehberi 

Bu rehber, CVAT'ı Docker ve Docker Compose kullanarak yerel olarak kurmak için adım adım talimatlar içerir.
> ⚠️ <span style="color:red">Bunu bir bilgisayarda kurup daha sonra aynı adres ile diğer bilgisayarlardan erişilebilir.</span>


## Gereksinimler

- Docker (v20.10+)
- Docker Compose (v2.0+)
- Git
- En az 8GB RAM
- En az 20GB disk alanı

## Kurulum Adımları

### 1. Sistemi Hazırlama

```bash
# Docker ve Docker Compose'un yüklü olduğunu kontrol edin
docker --version
docker compose version

# Gerekirse Docker'ı başlatın
sudo systemctl start docker
sudo systemctl enable docker

# Kullanıcınızı docker grubuna ekleyin (yeniden giriş gerekir)  Yetkisi olan kişi tarafından yapılır.
sudo usermod -aG docker $USER
```

### 2. CVAT Repository'sini Klonlama

```bash
# Ana dizinize gidin
cd ~

# CVAT repository'sini klonlayın
git clone https://github.com/cvat-ai/cvat.git
cd cvat

# Stable sürüme geçin (isteğe bağlı)
git checkout v2.7.6
```

### 3. Çevre Değişkenlerini Yapılandırma

Yerel IP adresinizi öğrenin:
```bash
# Yerel IP adresinizi bulun
ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d/ -f1
```

`terminale girilecek bash komutu`
`.env` dosyası oluşturun:   
```bash
# .env dosyasını oluşturun
cat > .env << EOF
CVAT_HOST=10.40.3.19
CVAT_VERSION=v2.7.6
COMPOSE_PROJECT_NAME=cvat
EOF
```

> **Not:** `10.40.3.19` kısmını kendi IP adresinizle değiştirin.

### 4. Port Yapılandırması

Eğer varsayılan port (8080) çakışıyorsa, `docker-compose.yml` dosyasını düzenleyin:

```bash
# Traefik servisindeki ports bölümünü bulun ve değiştirin
nano docker-compose.yml
```

Traefik ports bölümünü şu şekilde değiştirin: bu portlar kullnımdaysa değiştirilebilir.
```yaml
ports:
  - 8083:8080  # Ana CVAT portu
  - 8092:8090  # Traefik dashboard portu
```

### 5. CVAT'ı Başlatma

```bash
# Konteynerleri başlatın
docker compose up -d

# Başlatma durumunu kontrol edin
docker compose ps
```

Tüm konteynerler "Up" durumunda olmalı:
```
NAME                            STATUS
cvat_clickhouse                 Up
cvat_db                         Up
cvat_grafana                    Up
cvat_opa                        Up
cvat_redis                      Up
cvat_server                     Up
cvat_ui                         Up
traefik                         Up
```

### 6. Süper Kullanıcı Oluşturma / direkt adrese girip ui üzerinden kullanıcı oluşturabilirsiniz.

```bash
# CVAT için admin kullanıcı oluşturun
docker exec -it cvat_server python3 manage.py createsuperuser
```

Sistem sizden şunları isteyecek:
- **Username:** Admin kullanıcı adı
- **Email address:** E-posta adresi (isteğe bağlı)
- **Password:** Güçlü bir şifre
- **Password (again):** Şifreyi tekrar girin

### 7. CVAT'a Erişim

Tarayıcınızda şu adresi açın:
```
http://YOUR_IP_ADDRESS:8083
```

Örnek:
```
http://10.40.3.19:8083
```

Oluşturduğunuz kullanıcı adı ve şifre ile giriş yapın.

## Ek Ayarlar ve İpuçları

### Firewall Ayarları

Ubuntu/Debian için:
```bash
# 8083 portunu açın
sudo ufw allow 8083

# Firewall durumunu kontrol edin
sudo ufw status
```

### Logları Kontrol Etme

```bash
# Tüm servislerin loglarını görün
docker compose logs

# Belirli bir servisin loglarını görün
docker compose logs cvat_server
docker compose logs traefik
```

### CVAT'ı Durdurma ve Başlatma

```bash
# CVAT'ı durdurun
docker compose down

# CVAT'ı başlatın
docker compose up -d

# Konteynerleri tamamen kaldırın (veriler korunur)
docker compose down --remove-orphans
```

### Veri Yedekleme

```bash
# Veritabanı yedeği alın
docker exec cvat_db pg_dump -U root cvat > cvat_backup.sql

# Docker volume'ları listeleyin
docker volume ls | grep cvat
```

### Güncelleme

```bash
# Yeni sürüme güncelleme
git pull origin master
docker compose pull
docker compose up -d --build
```

## Sorun Giderme

### Port Çakışması

Eğer port kullanımda hatası alırsanız:
```bash
# Hangi uygulamanın portu kullandığını bulun
sudo netstat -tlnp | grep :8083
sudo lsof -i :8083

# Portu kullanan uygulamayı durdurun veya farklı port kullanın
```

### Konteyner Başlatma Sorunları

```bash
# Detaylı hata loglarını görün
docker compose logs --tail=50

# Belirli bir konteynerin durumunu kontrol edin
docker inspect cvat_server
```

### Disk Alanı Sorunu

```bash
# Kullanılmayan Docker objelerini temizleyin
docker system prune -a

# Docker volume'larını temizleyin (DİKKAT: Veri kaybı olabilir)
docker volume prune
```

### Ağ Bağlantı Sorunları

```bash
# Docker ağlarını listeleyin
docker network ls

# CVAT ağını yeniden oluşturun
docker compose down
docker network rm cvat_cvat
docker compose up -d
```

## Güvenlik Önerileri

1. **Güçlü şifreler kullanın** - Admin hesabı için karmaşık şifre seçin
2. **Firewall yapılandırın** - Sadece gerekli portları açık tutun
3. **Düzenli güncelleyin** - CVAT'ı güncel tutun
4. **Yedek alın** - Düzenli veri yedeği yapın
5. **HTTPS kullanın** - Prodüksiyon ortamında SSL sertifikası ekleyin

## Faydalı Komutlar

```bash
# CVAT durumunu kontrol et
docker compose ps

# Kaynak kullanımını görün
docker stats

# Konteyner içine girin
docker exec -it cvat_server bash

# Veritabanına bağlan
docker exec -it cvat_db psql -U root -d cvat

# CVAT ayarlarını görün
docker exec cvat_server python3 manage.py diffsettings
```

## Destek ve Kaynaklar

- **Resmi Dokümantasyon:** https://opencv.github.io/cvat/
- **GitHub Repository:** https://github.com/cvat-ai/cvat
- **Community Forum:** https://github.com/cvat-ai/cvat/discussions
- **Issue Tracker:** https://github.com/cvat-ai/cvat/issues

---

**Kurulum tamamlandıktan sonra CVAT'ı kullanmaya başlayabilirsiniz! İlk projenizi oluşturun ve görüntü etiketleme işlemine başlayın.**
