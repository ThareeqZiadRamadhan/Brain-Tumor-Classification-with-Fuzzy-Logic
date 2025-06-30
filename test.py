import pandas as pd
import numpy as np
from tqdm import tqdm

# === CONFIG ===
label_file = "bt_dataset_t3.csv"  # Gantilah dengan path file CSV yang sesuai

# === 1. Load Label CSV ===
df_label = pd.read_csv(label_file)

# Pastikan CSV terbaca dengan benar
print("CSV Data (Beberapa Baris Pertama):")
print(df_label.head())  # Cek apakah kolom 'Target' ada

# Cek nama kolom yang tersedia
print("Nama Kolom dalam CSV:", df_label.columns)  # Menampilkan nama kolom

labels = []

# === 2. Ekstrak Label dari CSV ===
# Memastikan tidak ada nilai nan pada kolom 'Target'
df_label = df_label.dropna(subset=['Target'])  # Menghapus baris yang memiliki nilai nan pada kolom 'Target'

# Mengekstrak label dari kolom 'Target'
for _, row in tqdm(df_label.iterrows(), total=len(df_label)):
    labels.append(row['Target'])  # Menggunakan kolom 'Target' sebagai label

# Memastikan jumlah data labels
print(f"Total jumlah data labels: {len(labels)}")

# Cek beberapa label pertama
print("Labels (10 Pertama):", labels[:10])  # Periksa apakah labels terisi dengan benar

# === 3. Encode Label ===
label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
print("Label Map:", label_map)  # Cek hasil mapping

# Encode labels menjadi numerik
y = np.array([label_map[label] for label in labels])
print("y (Encoded Labels):", y[:10])  # Pastikan 'y' sudah terisi