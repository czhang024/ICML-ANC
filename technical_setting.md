## Some Technical Settings

### Stable Mode
While the model works pretty well in the current setting, it can occasionally lead to errors for certain architectures. It is recommended to use the stable model by adding a normalization before control:
```bash
python main.py --stable_mode
```
### Learning Rate Decay
For datasets like SVHN and Food101, it is recommended to use a smaller lr decay:
```bash
python main.py --lr_decay=0.95
```