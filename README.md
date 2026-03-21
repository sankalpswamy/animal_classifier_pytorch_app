# Animal Classifier

This version is compatible with environments where TensorFlow is not available, including newer Python installations.

## 1) Put your selected folders into `data/raw/`
Use these 15 classes:
- antelope
- bat
- bear
- bee
- beetle
- bison
- boar
- butterfly
- camel
- capybara
- cat
- dog
- elephant
- fox
- zebra

## 2) Install dependencies
```bash
python -m pip install -r requirements.txt
```

## 3) Split the dataset
```bash
python training/split_dataset.py --config configs/default.yaml
```

## 4) Train the model
```bash
python training/train.py --config configs/default.yaml
```

## 5) Launch the app
```bash
python app/app.py
```

## Notes
- The training script uses EfficientNet-B0 transfer learning.
- It runs multiple trials and saves the best model to `models/best_model.pt`.
- Accuracy depends on dataset quality and class balance.
