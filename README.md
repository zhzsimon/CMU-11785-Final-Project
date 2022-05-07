## Application of Deep Reinforcement Learning in Stock Market Trading

This repository contains the final project of Team 5 for 11785 - Intro to Deep Learning. There are 5 stocks data in this repository, and one .ipynb notebook for running the code.

### Data Formation:

For each stock data, it contains the data from the time it went public in the stock market till the most recent data available. There are 9 features in total:

1. **Open**: open price of the stock
2. **High**: highest price on a day
3. **Low**: lowest price on a day
4. **Volume**: trading volume on a day
5. **Adj Close**: close price of the stock adjusted to divident payment, stock split, etc...
6. **VIX**: volatity of the market, reflecting the market sentiment on fear and uncertainty
7. **Nasdaq**: stock market index 
8. **WTI**: crude oil price
9. **LSTM Predicted**: predicted price by LSTM

### Running Instructions:

The code is made ready-to-go. Before you run, please make sure you have every dependencies installed. 

#### Load stock data:

To train the model on a specific stock, change file name in the following code:

```python
# load data
raw_data = pd.read_csv('AAPL.csv')
n_features = len(raw_data.columns) - 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LSTMNetwork(n_features).to(device)
print(model)
```

#### Select a period for training and validation:

To change the amount of data used for training and validation, modify the ratio in the variable **start** or **end**, or you can create your own variable, and pass it into the dataloader slicing.  

In LSTM:

```python
start = int(len(combined_scaled_data) * 0.4) # for baseline
end = int(len(combined_scaled_data) * 0.1) # for popular stocks like Amazon and APPLE

# Train Data Loading...
train_data = LibriSamples(combined_scaled_data.values[:-end, :-3], lookback_period, prediction_period)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_loader_no_shuffle = DataLoader(train_data, batch_size=batch_size, shuffle=False)

# Val Data Loading...
val_data = TestLibriSamples(combined_scaled_data.values[-end:, :-3], lookback_period)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                         shuffle=False)
```

In DQN:

```python
# for training
num_col = len(combined_scaled_data.columns) # number of features
env = stock(combined_scaled_data.iloc[:-end, :], combined_raw_data['Adj Close'].iloc[:-end], 
            num_col, lookback_period = LOOKBACK_PERIODS)
agent = Agent(env.n_features)
```

```python
# for validation
def BackTest():
    env_val = stock(combined_scaled_data.iloc[-end:, :].reset_index(drop=True), 
                    combined_raw_data['Adj Close'].iloc[-end:].reset_index(drop=True), 
                    num_col, lookback_period = LOOKBACK_PERIODS)
    observation = env_val.reset()
```

**If you are modifying above slicings, please make sure also to change the slicing in LSTM drawing.**

```python
import matplotlib.pyplot as plt
inversed_result_train = [(price * (closed_max - closed_min) + closed_min) for price in flat_result_train]
actual_price = combined_raw_data['Adj Close'].values[:-end].tolist()
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(inversed_result_train[:], 'r', label='pred')
plt.plot(actual_price, 'b', label='actual')
plt.show()
```

#### Test different combination of features:

You can slice on combined_scaled_data like the following:

```python
# slice by index
combined_scaled_data.iloc[-end:, :-1] # exclude last feature
combined_scaled_data.iloc[-end:, 4:] # include only features after 4 columns

# select by column names
combined_raw_data[['High', 'Open', 'Low', 'Adj Close']]
```

#### Run the validation:

```python
env_val = BackTest() # run on the validation set
env_val.draw('trade1.png', 'profit1.png') # draw the result
```

#### Check the stock's ROI and Sharpe Ratio (buy at the begining then hold, no sell):

```python
# Note to change the slicing accordingly
bh = combined_raw_data['Adj Close'].iloc[:start].reset_index(drop=True).values
bh_roi = bh[-1] / bh[0] - 1
bh_daily_ret = []
for i in range(len(bh) - 1):
    bh_daily_ret.append(bh[i+1] / bh[i] - 1)
bh_sharpe = np.mean(bh_daily_ret) / np.std(bh_daily_ret) * (252 ** 0.5)
bh_sharpe, bh_roi
```

### Credits:

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

https://link.springer.com/article/10.1007/s00607-019-00773-w

https://sg.finance.yahoo.com/



