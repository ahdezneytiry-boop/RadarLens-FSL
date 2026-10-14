class Config:
    # 数据
    num_classes = 2
    seq_len = 30
    radar_dim = 16
    img_size = 64

    # 训练
    batch_size = 8
    lr = 1e-3
    epochs = 10

    # 模型
    tau = 0.7

    # 路径
    save_path = "./checkpoints/model.pth"
