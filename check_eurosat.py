import os

EUROSAT_PATH = r"C:\Users\Adminn\New folder (4)\eurosat\EuroSAT_RGB"

print("Dataset path:", EUROSAT_PATH)

classes = sorted([
    folder for folder in os.listdir(EUROSAT_PATH)
    if os.path.isdir(os.path.join(EUROSAT_PATH, folder))
])

print("\nClasses:")
for c in classes:
    print(c)

print("\nNumber of classes:", len(classes))

print("\nNumber of images:")

total = 0

for c in classes:
    folder = os.path.join(EUROSAT_PATH, c)

    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(c, ":", count)

    total += count

print("\nTotal images:", total)