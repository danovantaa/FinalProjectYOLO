# Definisi variabel :

- c1 = Channel Input
- c2 = Channel Output
- e = expansion ratio (yang akan dikalikan dengan output untuk hidden channel pada bottleneck)
- k = kernel
- s = stride
- p = padding
- act = Aktivasi
- c\_ = hidden channel
- g = Groups.
- d = Dilation.
- dim=1 : dalam channel

# Definisi variabel :
- Autopad = hitung padding otomatis agar ukuran output konvolusi = ukuran input (same padding).
- 

# Backbone

## Convolution

**Fungsi Kegunaan**

- untuk mengekstraksi fitur dari gambar

**Struktur:**  
Conv2d -> BatchNorm2d -> SiLU

**Penjelasan:**

- **Conv2d** = Mengambil fitur dasar seperti garis, tepi, dan sudut.
- **BatchNorm2d** = Menormalkan nilai fitur dalam 1 batch agar training lebih stabil.
- **SiLU (Sigmoid Linear Unit)** = Membuat model belajar pola non-linear dengan lebih halus dan stabil.

## Bottleneck

**Konsep**

- Mengurangi jumlah channel sebelum diproses, lalu mengembalikannya lagi, sehingga komputasi lebih ringan tapi tetap kaya fitur.

**Struktur:**

- Convolution -> Convolution -> Shortcut

**Penjelasan:**

- Menggunakan dua convolution kecil untuk memproses fitur.
- Shortcut (residual connection) menjaga informasi penting agar tidak hilang.

## C2f

**Konsep**

- Mengurangi komputasi dengan memisahkan sebagian fitur, memproses subset-nya secara mendalam, lalu menggabungkannya kembali untuk menghasilkan representasi fitur yang kaya dan stabil.

**Struktur:**  
Convolution -> Split -> Bottleneck -> Concat -> Convolution

**Penjelasan:**

- **Split** = Fitur dari convolution di-split menjadi 2 branch:

  1. lewat bottleneck,
  2. lewat shortcut langsung.  
     Tujuannya agar model belajar cepat dan tidak kehilangan informasi awal.

- **Bottleneck** = Fitur dipadatkan dan diproses dengan dua convolution kecil.

- **Concat** = Menggabungkan semua fitur dari bottleneck dan shortcut untuk menghasilkan representasi fitur yang lebih informatif.

## SPPF (Spatial Pyramid Pooling Fast)

**Konsep**

- Menangkap informasi konteks dari berbagai skala objek melalui pooling bertingkat secara sangat efisien, sehingga model dapat memahami objek kecil, sedang, dan besar secara bersamaan.

**Struktur:**  
Convolution -> MaxPool2d -> MaxPool2d -> MaxPool2d -> Concat -> Convolution

**Penjelasan:**

- **MaxPool2d** = Mengambil nilai terbesar dari area kecil,  
  sehingga memperkuat kemampuan model dalam mendeteksi objek dengan berbagai ukuran.

## Neck

Upsample =

## Head

Detect = Convolution -> Convolution -> Conv2d -> Bbox Loss & Cls Loss (
Bbox Loss =
Cls Loss =
)

# YOLO V11 :

## C3K

**Konsep**
Mirip Seperti C2F Pada Version 8, namun tidak dilakukan split

- **Struktur:**  
  Convolution -> Bottleneck -> Concat -> Convolution

## C3K2

**Konsep**

-

**Struktur:**  
Convolution -> C3K -> Concat -> Convolution

**Penjelasan:**


# YOLO V12 :
