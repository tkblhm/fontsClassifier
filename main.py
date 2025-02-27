from nn import *

# Split dataset
dataset = FontDataset(DATASET_PATH, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training Samples: {train_size}, Validation Samples: {val_size}")
trainer = Trainer(model, device, train_loader, val_loader, criterion, optimizer)
trainer.load_model("weights.pth")
# trainer.predict(r"C:\Users\hxtx1\Pictures\Screenshots\屏幕截图 2025-02-26 133458.png")
# trainer.predict(f"D:\\gyt\\font_dataset\\BrushScript\\0-1.png")
# acc = 0
# for i in range(20):
#     font = FONT_CLASSES[i]
#     for j in range(100):
#         result = trainer.predict(f"D:\\gyt\\font_dataset\\{font}\\0-{j}.png")
#         acc += (result == i)
# print("acc:", acc)
trainer.train_model(epochs=5)
trainer.save_model("weights.pth")

