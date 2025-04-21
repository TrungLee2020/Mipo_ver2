import pandas as pd
import os
import hashlib
import datetime
import uuid


def create_document_metadata(base_folder, output_csv):
    """Tạo file metadata từ thư mục chứa tài liệu"""
    metadata = []

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            # Bỏ qua các file không phải tài liệu
            if not file.lower().endswith(('.pdf', '.docx', '.md', '.csv', 'xlsx', 'txt')):
                continue

            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, base_folder)

            # Tạo document_id từ đường dẫn
            doc_id = hashlib.md5(relative_path.encode()).hexdigest()[:12]

            # Lấy thông tin file
            file_size = os.path.getsize(filepath)
            modified_date = datetime.datetime.fromtimestamp(
                os.path.getmtime(filepath)
            ).strftime('%Y-%m-%d')

            # Lấy category từ thư mục cha
            category = os.path.basename(os.path.dirname(filepath))

            # Tạo mẫu metadata
            doc_metadata = {
                'document_id': doc_id,
                'filename': file,
                'title': os.path.splitext(file)[0],  # Mặc định lấy tên file là title
                'filepath': relative_path,
                'date': '',
                'category': category,
                'author': '',
                'source': '',
                'description': '',
            }

            metadata.append(doc_metadata)

    # Tạo DataFrame và lưu vào CSV
    df = pd.DataFrame(metadata)
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Đã tạo metadata cho {len(metadata)} tài liệu và lưu vào {output_csv}")
    return df


create_document_metadata("../data_sample/DTPT_VNPOST", "metadata.csv")