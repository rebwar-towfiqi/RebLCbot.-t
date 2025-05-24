import os

def find_and_delete_db(root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        for file in filenames:
            if file == "users.db":
                full_path = os.path.join(dirpath, file)
                print(f"✅ پیدا شد: {full_path} — در حال حذف...")
                os.remove(full_path)
                print("🗑️ حذف شد.")
                return
    print("❌ فایل users.db پیدا نشد.")

# مسیر اصلی پروژه خود را در اینجا مشخص کنید
find_and_delete_db("C:/Users/rebwa/RebLCbot")
