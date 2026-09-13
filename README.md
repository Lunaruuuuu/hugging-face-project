# hugging-face-project

## Model

Model yang digunakan:

**Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis**

Link Hugging Face:  
https://huggingface.co/Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis

Model digunakan untuk melakukan **sentiment analysis** pada teks berbahasa Indonesia dengan tiga kategori:
- Positive
- Neutral
- Negative

## Requirements

- Python 3.11
- PyTorch
- Transformers

## Instalasi

Clone repository:

- Buka Terminal lalu gunakan command berikut
git clone https://github.com/Lunaruuuuu/hugging-face-project.git
cd hugging-face-project

- Lalu pastikan anda sedang ada di folder project
jika belum gunakan cd <path folder project>

- Gunakan virtual environment
command :
python -m venv .venv
.venv\Scripts\activate

- Install dependencies
command : 
pip install -r requirements.txt

-Lalu jalankan program
command :
python sentiment_analysis.py
