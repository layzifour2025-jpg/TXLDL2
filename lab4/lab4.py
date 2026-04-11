import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import re

# =====================================================================
# THIẾT LẬP THÔNG SỐ CHO TỪNG BÀI
# (Bạn thay đổi các biến này tương ứng với từng file CSV)
# =====================================================================
FILE_PATH = 'ITA105_Lab_4_Hotel_reviews.csv' # Đổi tên file tương ứng Bài 1,2,3,4
TEXT_COLUMN = 'Noi_Dung_Review'              # Đổi tên cột chứa văn bản trong CSV
CATEGORICAL_COLUMNS = ['Loai_Phong']         # Đổi tên các cột cần Label/One-hot encoding
TARGET_WORD = 'sạch_sẽ'                      # Từ khóa cần tìm: Bài 1: sạch_sẽ, Bài 2: xuất_sắc, Bài 3: đẹp, Bài 4: sáng_tạo

# =====================================================================
# 1. NẠP DỮ LIỆU & KIỂM TRA MISSING VALUES
# =====================================================================
df = pd.read_csv(FILE_PATH)
print("--- KIỂM TRA DỮ LIỆU THIẾU ---")
print(df.isnull().sum())

# Xử lý missing values (Loại bỏ các dòng bị rỗng)
df = df.dropna().reset_index(drop=True)

# =====================================================================
# 2. LABEL ENCODING CÁC BIẾN CATEGORICAL
# =====================================================================
label_encoder = LabelEncoder()
for col in CATEGORICAL_COLUMNS:
    # Cột mới sẽ có hậu tố _encoded
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])
print("\n--- DỮ LIỆU SAU KHI ENCODING ---")
print(df.head(3))

# =====================================================================
# 3. TIỀN XỬ LÝ VÀN BẢN (Lowercase, Tokenization, Stop words)
# =====================================================================
# Danh sách stop words tiếng Việt cơ bản (Bạn có thể bổ sung thêm)
stop_words = ['là', 'và', 'của', 'có', 'để', 'với', 'những', 'các', 'trong', 'cho']

def preprocess_text(text):
    # Lowercase
    text = str(text).lower()
    # Loại bỏ dấu câu cơ bản
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization (Tách từ đơn giản bằng khoảng trắng)
    # Lưu ý: Với tiếng Việt chuẩn, nên dùng word_tokenize của underthesea để nối từ ghép (vd: sạch_sẽ)
    tokens = text.split()
    # Nếu dùng tách từ đơn giản, ta ghép từ khóa thủ công để Word2Vec tìm được (vd: 'sạch', 'sẽ' -> 'sạch_sẽ')
    # Ở đây giả định text đã được chuẩn hóa hoặc ta tìm từ đơn.
    
    # Loại bỏ stop words
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df['processed_tokens'] = df[TEXT_COLUMN].apply(preprocess_text)
# Tạo lại câu string từ tokens để dùng cho TF-IDF
df['processed_text'] = df['processed_tokens'].apply(lambda x: ' '.join(x))

# =====================================================================
# 4. TẠO TF-IDF MATRIX
# =====================================================================
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
print("\n--- KÍCH THƯỚC MA TRẬN TF-IDF ---")
print(tfidf_matrix.shape)

# =====================================================================
# 5. TẠO WORD2VEC EMBEDDINGS & TÌM TỪ GẦN NGHĨA
# =====================================================================
# Huấn luyện mô hình Word2Vec
sentences = df['processed_tokens'].tolist()
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

print(f"\n--- 5 TỪ GẦN NGHĨA VỚI '{TARGET_WORD}' ---")
try:
    # Lưu ý: Nếu TARGET_WORD là từ ghép (sạch_sẽ), thuật toán tokenization ở bước 3 
    # phải nối được từ đó, nếu không model sẽ báo lỗi KeyError.
    similar_words = w2v_model.wv.most_similar(TARGET_WORD, topn=5)
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")
except KeyError:
    print(f"Từ '{TARGET_WORD}' không xuất hiện trong từ điển. Vui lòng kiểm tra lại quá trình tách từ (Tokenization).")