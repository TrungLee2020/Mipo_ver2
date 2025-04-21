import fitz  # PyMuPDF
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(log_name="mipo",
                 log_level=logging.INFO,
                 log_file="mipo.log",
                 console_output=True,
                 log_dir="logs"):
    """
    Thiết lập logger cho ứng dụng với một file log duy nhất.

    Args:
        log_name (str): Tên của logger
        log_level (int): Mức độ logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str): Tên file log cố định
        console_output (bool): Hiển thị log trên console
        log_dir (str): Thư mục chứa file log

    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình
    """
    # Tạo logger
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # Đảm bảo logger không lặp lại handlers nếu được gọi nhiều lần
    if logger.handlers:
        return logger

    # Định dạng log
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Thêm log vào console nếu cần
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    # Luôn ghi log vào một file duy nhất
    log_directory = Path(log_dir)
    log_directory.mkdir(exist_ok=True, parents=True)

    file_path = log_directory / log_file
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """
    Lấy logger đã được cấu hình trước đó hoặc tạo mới nếu chưa tồn tại.

    Args:
        name (str, optional): Tên của module gọi logger. Mặc định sẽ lấy từ tên file.

    Returns:
        logging.Logger: Đối tượng logger
    """
    if name is None:
        # Lấy tên từ module gọi hàm này
        import inspect
        frm = inspect.stack()[1]
        module = inspect.getmodule(frm[0])
        name = module.__name__ if module else "unnamed"

    logger = logging.getLogger(name)

    # Nếu logger chưa được cấu hình, tạo một logger mặc định
    if not logger.handlers:
        return setup_logger(log_name=name)

    return logger


def log_function_call(func):
    """
    Decorator để ghi log khi một hàm được gọi và kết thúc.

    Args:
        func: Hàm cần ghi log

    Returns:
        wrapper: Hàm đã được bọc với tính năng ghi log
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Bắt đầu hàm {func.__name__} với tham số: args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Kết thúc hàm {func.__name__} thành công")
            return result
        except Exception as e:
            logger.error(f"Lỗi trong hàm {func.__name__}: {str(e)}", exc_info=True)
            raise

    return wrapper

def is_scanned_pdf(pdf_path):
    """
    Kiểm tra xem file PDF có phải là file scan hay không.

    Phương pháp:
    1. Kiểm tra xem PDF có chứa text hay không
    2. Kiểm tra xem PDF có chứa hình ảnh lớn hay không
    3. So sánh tỷ lệ giữa text và hình ảnh

    Returns:
        bool: True nếu PDF có khả năng là file scan, False nếu không
    """
    doc = fitz.open(pdf_path)

    total_text = 0
    total_images = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Lấy text thực sự từ trang
        text = page.get_text("text")
        text_length = len(text.strip())
        total_text += text_length

        # Lọc hình ảnh có kích thước đủ lớn
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width > 100 and pix.height > 100:
                    total_images += 1
                pix = None  # giải phóng bộ nhớ
            except:
                continue

        # Kiểm tra nhanh: ít text nhưng có ảnh lớn -> có thể là file scan
        if text_length < 10 and total_images > 0:
            doc.close()
            return True

    doc.close()

    # Tổng kết để xác định PDF scan
    if total_images > 0 and total_text < total_images * 100:
        return True

    if total_text == 0 and total_images > 0:
        return True

    return False
