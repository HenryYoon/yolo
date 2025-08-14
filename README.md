## YOLO

Custom Library to Train YOLO Model.

### Library Structure
```
yolo
├── coco128
│   ├── images
│   │   └── train2017
│   ├── labels
│   │   └── train2017
│   ├── LICENSE
│   └── README.txt
├── configs
│   ├── default.yaml
│   └── yolo.yaml
├── core
│   ├── base
│   │   └── trainer.py
│   ├── callbacks
│   └── register
├── data
│   ├── dataloaders
│   │   └── loaders.py
│   ├── datasets
│   └── transforms
│       ├── letterbox.py
│       └── mosaic.py
├── nn
│   ├── losses
│   │   ├── detect.py
│   │   └── segment.py
│   ├── models
│   └── modules
└── tests
    ├── data
    ├── models
    ├── predict
    └── train

```


### TODO:
- [ ] Implement Transforms
  - [x] LetterBox
  - [ ] Mosaic
  - [ ] Flip
  - [ ] HSV
  - [ ] Pass tests
- [ ] Implement Modules
  - [ ] Backbone
  - [ ] Neck
  - [ ] Head
  - [ ] Pass tests
- [ ] Implement Post-Process
  - [ ] Process Head Outputs
  - [ ] NMS
  - [ ] Pass tests
- [ ] Implement Miscellaneous
  - [ ] Register
    - [ ] Data
    - [ ] Model
  - [ ] Metrics
  - [ ] Pass tests
- [ ] Train/Predict Test on COCO128 dataset