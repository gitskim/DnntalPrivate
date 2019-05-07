from sklearn.model_selection import train_test_split

X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(X, X_feat, y, test_size=0.2, random_state=42)

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-dentist_AI', verbose=1, save_best_only=True, save_weights_only=True)
]


# To use the model 2
input_size = (img_row, img_col, img_chan)
sgd = SGD(lr=0.01, momentum=0.9)
model2 = unet(sgd, input_size, tversky_loss)


#Train the network
results = model2.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, validation_split=0.15, shuffle=True, callbacks=callbacks,
                    validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))

