def predict_dataframe(classification_model, data_frame):
            predictions = []
            for i in range(data_frame.shape[0]):
                predictions.append(classification_model.predict(data_frame.iloc[i]['text'])[0][0])
            groundtruth = data_frame['label'].tolist()
            return predictions, groundtruth