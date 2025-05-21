import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

CATEGORICAL = ['PULocationID', 'DOLocationID']
TARGET = "duration"

def read_dataframe(filename: str):
    data = pd.read_parquet(filename)

    if filename == "yellow_tripdata_2023-01.parquet":
        print("Columns in January data: %d" % len(data.columns))

    # Transforming pick-up and drop-off time columns into datetime data type
    data.tpep_pickup_datetime = pd.to_datetime(data.tpep_pickup_datetime)
    data.tpep_dropoff_datetime = pd.to_datetime(data.tpep_dropoff_datetime)

    # Creating duration column
    data["duration"] = data.tpep_dropoff_datetime-data.tpep_pickup_datetime

    # Converting duration to minutes
    data.duration = data.duration.apply(lambda dur: dur.total_seconds()/60)
    if filename == "yellow_tripdata_2023-01.parquet":
        print("Standard deviation of trip duration in January: %.2f" % data.duration.std())

    # Removing outliers (duration shorter than one minute or longer than sixty minutes)
    total_rows = len(data)
    data = data[(data.duration >= 1) & (data.duration <= 60)]
    if filename == "yellow_tripdata_2023-01.parquet":
        print("Remaining percentage of records after outlier removal: %d%%" % (100*len(data)/total_rows))

    # Preparing one hot encoding
    # Setting pick-up and drop-off locations types to string
    data[CATEGORICAL] = data[CATEGORICAL].astype(str)

    return data


def get_features(train_df, val_df):
    dv = DictVectorizer()

    x_train = dv.fit_transform(
        train_df[CATEGORICAL].to_dict(orient="records")
    )

    x_val = dv.transform(
        val_df[CATEGORICAL].to_dict(orient="records")
    )

    return x_train, x_val




if __name__ == "__main__":


    # Fitting DictVectorizer
    train_df = read_dataframe("yellow_tripdata_2023-01.parquet")
    val_df = read_dataframe("yellow_tripdata_2023-02.parquet")
    
    dv = DictVectorizer()
    x_train = dv.fit_transform(
        train_df[CATEGORICAL].to_dict(orient="records")
    )

    print("Dimension of feature matrix after: %d" % x_train.shape[1])

    # Preparing target
    y_train = train_df[TARGET].values

    # Preparing linear regression model
    lr = LinearRegression().fit(x_train, y_train)
    y_pred = lr.predict(x_train)
    print("Linear regression RMSE on train dataset: %.2f" % root_mean_squared_error(y_pred, y_train))   

    x_val = dv.transform(
        val_df[CATEGORICAL].to_dict(orient="records")
    )
    y_val = val_df[TARGET].values
    y_pred = lr.predict(x_val)
    print("Linear regression RMSE on val dataset: %.2f" % root_mean_squared_error(y_pred, y_val))
