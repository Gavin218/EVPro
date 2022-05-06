def dimReduction(dataPath, sheet, modelPath, lowDataSavePath):
    from Pretreatment import excel_to_matrix
    import tensorflow as tf
    import pickle
    dataset = excel_to_matrix(dataPath, sheet)
    total = len(dataset)
    length = len(dataset[0])
    model = tf.saved_model.load(modelPath)

    tfData = tf.reshape(dataset, [total, length])
    # tfData = tf.cast(tfData, dtype=tf.float32)
    lowDimData = model.decoder(tfData).numpy()
    with open(lowDataSavePath, 'wb') as f:
        pickle.dump(lowDimData, f)
    return 0


dataPath = "D:/Backup/桌面/EVC/loadData.xlsx"
sheet = 0
modelPath = "D:/Backup/桌面/EVC/AEmodel"
lowPath = "D:/Backup/桌面/EVC/lowDemData"
dimReduction(dataPath, sheet, modelPath, lowPath)