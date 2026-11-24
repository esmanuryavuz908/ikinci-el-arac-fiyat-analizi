# 🚗 İkinci El Araç Piyasası Analizi ve Fiyat Tahmini

İkinci el araç ilanlarını SQLite, Keşifsel Veri Analizi (EDA), istatistiksel hipotez testi, özellik mühendisliği (Feature Engineering) ve Doğrusal Regresyon modelleriyle uçtan uca analiz eden veri bilimi projesi.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🎯 Projenin Amacı ve Çözdüğü Problem

İkinci el araç piyasasında ilan fiyatları; marka algısı, üretim yılı, kilometre, vites tipi, çekiş sistemi ve hasar/ruhsat durumu (`title_status`) gibi çok sayıda değişkene bağlı olarak ciddi dalgalanmalar göstermektedir. Ayrıca ham ilan verileri eksik değerler, tekrarlayan kayıtlar ve hatalı/absürt fiyat girişleri (aykırı değerler) barındırmaktadır.

Bu proje, ham ikinci el araç ilan verisinden 10.000 satırlık bir örneklem üzerinde çalışarak:
- Veriyi ilişkisel bir SQLite veritabanına (`araclar.db`) aktarıp SQL sorgularıyla temel piyasa metriklerini çıkarmayı,
- Aykırı değer ve eksik verilerden arındırılmış temiz bir analiz veri seti oluşturmayı,
- Araç fiyatını etkileyen temel faktörleri görselleştirme ve istatistiksel testlerle doğrulamayı,
- Özellik mühendisliğinin Doğrusal Regresyon modelinin açıklayıcılık gücüne ($R^2$) etkisini ölçmeyi amaçlamaktadır.

---

## ✨ Öne Çıkan Özellikler ve Proje Bulguları

### Uygulanan Veri Bilimi Adımları
- **SQLite Entegrasyonu ve SQL Sorguları:** `vehicles.csv.zip` dosyasından okunan ilk 10.000 satır `araclar.db` veritabanındaki `vehicles` tablosuna aktarılarak; en pahalı 10 araç, marka bazlı ortalama fiyat, 2020 sonrası & 100.000 km altı araçlar, `prescott` bölgesindeki otomatik araç ortalaması ve marka bazlı ilan sayıları SQL ile sorgulandı.
- **Veri Temizleme ve Aykırı Değer Analizi (IQR):** Tekrarlayan satırlar kaldırıldı, `odometer` ve `year` eksikleri medyan ile dolduruldu, kritik sütunlarda (`price`, `manufacturer`, `model`) eksik değer içeren kayıtlar elendi. `price` ($500\$$ alt sınırı ile birlikte) ve `odometer` için IQR ($Q3 + 1.5 \times IQR$) yöntemiyle aşırı uç/hatalı ilanlar filtrelendi.
- **Olasılık ve Hipotez Testi:** Bir aracın ortalama fiyatın üzerinde olma ve 100.000 km altında olma olasılıkları hesaplandı. Otomatik ve manuel vitesli araç fiyatları arasında anlamlı fark olup olmadığı Welch bağımsız iki örneklem t-testi (`scipy.stats.ttest_ind`, `equal_var=False`, $\alpha = 0.05$) ile test edildi.
- **Özellik Mühendisliği (Feature Engineering):**
  - `car_age`: Araç yaşı (`2026 - year`, 0 yaş kayıtları 1 olarak düzenlendi)
  - `km_per_year`: Yıllık ortalama kullanım (`odometer / car_age`)
  - `is_4wd`: 4x4 çekiş bayrağı (`drive == '4wd'`)
  - `is_premium`: Premium marka bayrağı (`bmw`, `mercedes-benz`, `audi`, `volvo`, `lexus`, `porsche`)
  - `high_mileage`: 150.000 km üzeri araç bayrağı
  - `segment`: Kasa tiplerinin (`type`) **Binek**, **SUV**, **Ticari_ve_Buyuk**, **Diger** ve **Bilinmiyor** gruplarına indirgenmesi
- **Kodlama ve Modelleme:** Kategorik sütunlar (`transmission`, `fuel`, `manufacturer`, `title_status`, `segment`) One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`) ile sayısallaştırıldı; Basit ve Çoklu Doğrusal Regresyon (`LinearRegression`) modelleri karşılaştırıldı ve en yüksek hatalı 10 tahmin üzerinden hata analizi yapıldı.

### Temel Bulgular
- **Kilometre Etkisi (Basit Regresyon):** Tek değişkenli regresyon analizinde kilometre (`odometer`) katsayısı **`-0.11`** olarak hesaplandı; her 1 birimlik kilometre artışı araç fiyatında yaklaşık 11 cent düşüşle ilişkilendirildi.
- **Feature Engineering Katkısı (Model 1 vs. Model 2):**
 - **Model 1 (Ham Veriler - Feature Engineering Öncesi) $R^2$ Skoru:** `0.5651` (%56,51)
 - **Model 2 (Feature Engineering Sonrası Çoklu Regresyon) $R^2$ Skoru:** `0.6529` (%65,29) — *(Üretilen yeni özellikler model başarısını **+0.0878 puan** artırdı.)*
- **Fiyatı En Çok Etkileyen Faktörler:** Katsayı analizine göre fiyatı en çok artıran değişkenler **Tesla** (`+$17.398`), **Porsche** (`+$6.074`) ve **4x4 çekiş sistemi (`is_4wd`)** (`+$5.198`) olurken; en çok düşürenler **Fiat** (`-$13.640`), **elektrikli yakıt etiketi** (`-$11.257`) ve **eksik ruhsat durumu (`title_status_missing`)** (`-$9.927`) oldu.
- **Hasar / Ruhsat Durumu Etkisi:** `title_status` değeri `missing` (eksik) veya `salvage` (hurda/pert) olan ilanlarda fiyatın **10.000 dolara kadar** düştüğü gözlemlendi.
- **Hipotez Testi Sonucu (Otomatik vs. Manuel):** Welch t-testi sonucunda **`p-value = 0.1808`** hesaplandı. $p \ge 0.05$ olduğu için $H_0$ reddedilemedi; bu veri örnekleminde otomatik ve manuel vitesli araçların fiyatları arasında istatistiksel olarak anlamlı bir fark bulunmadı.

---

## 🛠️ Kullanılan Teknolojiler ve Araçlar

| Kategori | Teknoloji / Kütüphane | Kullanım Amacı |
| :--- | :--- | :--- |
| **Dil & Ortam** | Python 3.13, Jupyter Notebook | Analiz geliştirme ve etkileşimli raporlama |
| **Veritabanı** | SQLite (`sqlite3`) | İlan verisini ilişkisel tabloda saklama ve SQL sorgulama |
| **Veri İşleme** | Pandas, NumPy | Veri okuma, temizleme, özellik mühendisliği, encoding |
| **İstatistik** | SciPy (`scipy.stats`) | Welch bağımsız iki örneklem t-testi |
| **Görselleştirme** | Matplotlib, Seaborn | Histogram, Boxplot, Countplot, Scatterplot ve Korelasyon Isı Haritası |
| **Makine Öğrenmesi** | Scikit-learn (`LinearRegression`, `r2_score`) | Basit & Çoklu Doğrusal Regresyon modelleme ve $R^2$ değerlendirme |

---

## 📁 Proje Klasör ve Dosya Yapısı

```text
ikinci-el-arac-fiyat-analizi/
├── arac_analizi.ipynb      # SQL, EDA, istatistik, feature engineering ve regresyon notebook'u
├── .gitignore              # Büyük veri seti, veritabanı ve geçici Jupyter dosyalarını hariç tutar
├── README.md               # Proje dokümantasyonu
├── vehicles.csv.zip        # (Repoda yer almaz - yerel çalıştırma için indirilecek ham veri seti)
└── araclar.db              # (Repoda yer almaz - notebook çalıştırıldığında otomatik oluşur)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/esmanuryavuz908/ikinci-el-arac-fiyat-analizi.git
cd ikinci-el-arac-fiyat-analizi
```

### 2. Gerekli Kütüphaneleri Kurun
Sanal ortam veya Anaconda ortamınızda aşağıdaki paketlerin kurulu olduğundan emin olun:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn notebook
```

### 3. Veri Setini Hazırlayın
1. Kullanılan ikinci el araç veri setini (`vehicles.csv.zip`) şu kaynaktan indirin: [Kaggle - Used Cars Dataset (Craigslist)](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
2. İndirdiğiniz `vehicles.csv.zip` dosyasını proje ana klasörüne (yani `arac_analizi.ipynb` ile aynı dizine) yerleştirin.
3. Notebook içindeki `dosya_yolu` değişkeninin `"vehicles.csv.zip"` olarak ayarlı olduğundan emin olun.

### 4. Notebook'u Çalıştırın
```bash
jupyter notebook arac_analizi.ipynb
```
Açılan arayüzde **Kernel > Restart & Run All** adımıyla tüm analiz, SQLite veritabanı oluşturma (`araclar.db`) ve modelleme hücrelerini sırasıyla çalıştırabilirsiniz.

---
## 📝 Veri Seti ve Lisans Notu
- **Veri Seti Kaynağı:** [Kaggle - Used Cars Dataset (Craigslist)](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- **Geliştirici:** Veri Bilimi ve Analitiği Öğrencisi Esmanur Yavuz
