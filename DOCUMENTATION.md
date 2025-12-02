# 📚 Panduan Dokumentasi

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements-docs.txt
```

### Build & Serve Lokal

```bash
# Serve dengan auto-reload
mkdocs serve

# Buka browser di http://127.0.0.1:8000
```

### Build Static Site

```bash
# Build dokumentasi
mkdocs build

# Output akan di folder `site/`
```

## 🌐 Deploy ke GitHub Pages

### Setup Awal

1. **Update `mkdocs.yml`:**
   - Uncomment dan ganti `YOUR_USERNAME` dengan username GitHub Anda
   - Update `site_url` dan `repo_url`

2. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Source: GitHub Actions

3. **Push ke GitHub:**
   ```bash
   git add .
   git commit -m "Add documentation"
   git push origin main
   ```

### Auto-Deploy

Dokumentasi akan otomatis di-deploy ketika:
- Push ke branch `main` atau `master`
- File di folder `docs/` atau `mkdocs.yml` berubah
- GitHub Actions workflow berhasil dijalankan

### Manual Deploy

```bash
# Install mkdocs gh-deploy plugin
pip install mkdocs-material[imaging]

# Deploy
mkdocs gh-deploy
```

## 📁 Struktur Dokumentasi

```
docs/
├── index.md                    # Halaman utama
├── about/                      # Tentang proyek
├── dataset/                    # Dokumentasi dataset
├── methodology/               # Metodologi
├── experiments/               # Detail eksperimen
├── results/                    # Hasil eksperimen
├── usage/                      # Panduan penggunaan
├── conclusions/                # Kesimpulan
├── references.md              # Referensi
└── stylesheets/               # Custom CSS
    └── extra.css
```

## 🎨 Customization

### Theme Colors

Edit `mkdocs.yml`:

```yaml
theme:
  palette:
    - scheme: default
      primary: blue  # Ganti warna primary
      accent: blue   # Ganti warna accent
```

### Custom CSS

Edit `docs/stylesheets/extra.css` untuk custom styling.

### Navigation

Navigation otomatis di-generate dari struktur folder. Atau edit `nav:` di `mkdocs.yml` untuk custom navigation.

## 🔧 Troubleshooting

### Build Error

```bash
# Clear cache
rm -rf .cache site/

# Rebuild
mkdocs build --clean
```

### GitHub Pages Not Updating

1. Check GitHub Actions workflow status
2. Verify `site_url` di `mkdocs.yml`
3. Check repository Settings → Pages

### Local Server Not Starting

```bash
# Check port availability
lsof -i :8000

# Use different port
mkdocs serve -a 127.0.0.1:8080
```

## 📖 Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages](https://pages.github.com/)

## 💡 Tips

1. **Preview sebelum commit:** Selalu test lokal dengan `mkdocs serve`
2. **Check links:** Gunakan `mkdocs build --strict` untuk check broken links
3. **Version control:** Commit perubahan ke `docs/` dan `mkdocs.yml`
4. **Backup:** Backup dokumentasi sebelum perubahan besar

---

**Happy Documenting! 📚**

