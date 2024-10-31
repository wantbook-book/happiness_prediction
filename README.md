# Happiness Prediction

## install

python==3.11

```python
pip install -r requirements.txt
```


## start

1. preprocess data

    `python process_raw_data.py`

    ```python
    main(
        drop_unlabled=False,
        filtered_features_thres=0.005,
        raw_data_dir = src_dir/'dataset/raw_data',
        processed_data_dir = src_dir/'dataset/processed_data',
        for_change_filtered_features_thres=False
    )
    ```

    - `drop_unlabeled`: 部分happiness字段为-8，选择丢弃还是填充为3
    - `filtered_features_thres`：过滤与happiness低相关性的字段阈值
    - `raw_data_dir`：未处理的原始数据目录
    - `processed_data_dir`：处理后数据保存目录
    - `for_change_filtered_features_thres`：如果需要生成多个相关性过滤阈值的数据集而不重新生成其他数据集，在修改`filtered_features_thres`后将该参数也设为`True`

2. 算法

    ```bash
    python baseline_and_advanced_code.py
    python improved_code.py
    ```
    - `baseline_and_advanced_code.py`: 基线及五种进阶方法的代码，设置好数据集位置后，可直接于colab平台运行，或安装好相关包后于本机运行。
    - 'improved_code.py': 最终优化方法的代码，设置好数据集位置后，可直接于colab平台运行，或安装好相关包后于本机运行。

    

    

