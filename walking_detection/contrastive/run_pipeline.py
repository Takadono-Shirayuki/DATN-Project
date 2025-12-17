import subprocess
import sys
import time
import os

def run_script(script_name):
    """Hàm chạy một file python con và chờ nó kết thúc"""
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG CHẠY: {script_name}")
    print(f"{'='*50}\n")
    
    # Lấy đường dẫn trình thông dịch Python đang dùng hiện tại
    python_executable = sys.executable 
    
    start_time = time.time()
    
    # Gọi lệnh hệ thống để chạy file
    try:
        # check=True sẽ ném lỗi nếu file con chạy bị lỗi
        subprocess.run([python_executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ LỖI: Script {script_name} bị crash!")
        print("Dừng pipeline.")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"\n✅ HOÀN THÀNH {script_name} trong {duration:.1f} giây.")

if __name__ == "__main__":
    print("BẮT ĐẦU PIPELINE HUẤN LUYỆN TOÀN DIỆN")
    
    # Bước 1: Kiểm tra xem file common.py có tồn tại không
    if not os.path.exists("common.py"):
        print("❌ Lỗi: Không tìm thấy 'common.py'. Vui lòng kiểm tra lại.")
        sys.exit(1)

    # Bước 2: Chạy Pre-training (Giai đoạn 1)
    run_script("pretrain.py")
    
    # Bước 3: Chạy Classification (Giai đoạn 2)
    # File này sẽ tự động load encoder mà bước 1 vừa lưu
    run_script("train.py")
    
    print("\n" + "="*50)
    print("🎉 TẤT CẢ ĐÃ XONG! BẠN CÓ THỂ ĐI NGỦ NGON.")
    print("="*50)