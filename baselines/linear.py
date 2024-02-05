from sklearn.linear_model import LinearRegression

# linear_without_indoor_temp
def model_train_linear(train_current_std, feature_std_cols, label_std_col):

    x_train = train_current_std[feature_std_cols]
    y_train = train_current_std[label_std_col]
    model = LinearRegression()
    model.fit(x_train, y_train)

    print(model.coef_)
    # print('coef_')

    return model