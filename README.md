# Deneme-003

Streamlit tabanlı veri analiz asistanı.

## Kurulum

Projeyi klonladıktan sonra gerekli paketleri yükleyin:

```bash
pip install streamlit pandas numpy plotly openpyxl pyarrow scipy requests python-dotenv fastapi uvicorn
```

## Ortam Değişkenleri

LLM tabanlı analiz bölümünü kullanmak için OpenAI API anahtarınızı `.env` dosyasında tanımlayın. Uygulama başlangıçta bu dosyayı otomatik olarak yükler.

1. Proje dizinine bir `.env` dosyası oluşturun.
2. Dosyaya aşağıdaki satırı ekleyin:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```
3. Dosyayı kaydettikten sonra uygulamayı yeniden başlatın.

`.env` dosyasını sürüm kontrolünden hariç tutmak için proje içinde `.gitignore` dosyasında listelendiğinden emin olun.

## Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

### FastAPI Arka Uç

Uygulamanın basit bir HTTP arayüzünü başlatmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Bu komut FastAPI tabanlı arka uç servisini başlatır ve `/` ile `/health` uç noktalarından durum bilgisi sağlar.

## Notlar

- Büyük CSV dosyaları yüklerken doğru karakter kodlamasını seçtiğinizden emin olun.
- OpenAI API yanıtları ücretlendirilir; kullanım limitlerinizi kontrol edin.
