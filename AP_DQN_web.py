import pandas as pd
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import base64
from io import BytesIO
import quantstats as qs
from matplotlib.ticker import MaxNLocator

def clean_data(file_path, data_split):
    ohlcv_data = pd.read_csv(file_path).reset_index(drop=True)
    column_names_mapping = {'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    desired_order = ['date', 'open', 'high', 'low', 'close', 'volume']
    ohlcv_data = ohlcv_data.rename(columns=column_names_mapping)[desired_order]
    ohlcv_data.dropna(inplace=True)
    ohlcv_data['date'] = pd.to_datetime(ohlcv_data['date'], format='%Y-%m-%d')
    ohlcv_data['volume'] = ohlcv_data['volume'].astype(np.int32)
    ohlcv_data['log_diff'] = np.log(ohlcv_data['close'] / ohlcv_data['close'].shift(1))

    # Remove outliers outside 2 standard deviations
    num_std = 2
    for column in ['volume', 'log_diff']:
        mean = ohlcv_data[column].mean()
        std = ohlcv_data[column].std()
        upper_limit = mean + (num_std * std)
        lower_limit = mean - (num_std * std)
        ohlcv_data = ohlcv_data[(ohlcv_data[column] <= upper_limit) & (ohlcv_data[column] >= lower_limit)]

    for column in ['open', 'high', 'low', 'volume']: #I keep the close price not adjusted for simplicity#
        #normalize the data using min max normalization in columns
        ohlcv_data[column] = (ohlcv_data[column] - ohlcv_data[column].min()) / (ohlcv_data[column].max() - ohlcv_data[column].min())

    ohlcv_data.reset_index(drop=True, inplace=True)

    # Splitting the dataset into training and testing sets
    split_index = int(len(ohlcv_data) * (data_split/100))
    ohlcv_data_train = ohlcv_data[:split_index]
    ohlcv_data_test = ohlcv_data[split_index:]

    return ohlcv_data, ohlcv_data_train, ohlcv_data_test

class ContinuousOHLCVEnv(gym.Env):
    def __init__(self, ohlcv_data: pd.DataFrame, initial_cash: int =10000):
        self.ohlcv_data = ohlcv_data
        self.initial_cash = initial_cash
        self.available_actions = (0,1,2) #DONT THINK ITS BEING USED 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(ohlcv_data.shape[1],))
        self.max_idx = ohlcv_data.shape[0] - 1
        self.finish_idx = self.max_idx # New
        self.start_idx = 0 # New
        self.purchase_history = [] #NEWLY ADDED TO STORE PRICE AT WHICH BOUGHT
        self.reset()

    def reset(self):
        self.current_step = self.start_idx
        self.cash_in_hand = self.initial_cash
        self.stock_holding = int(0)
        self.step_info = []  # Initialize an empty list to store step information
        self.stock_price = self.ohlcv_data[self.current_step][3]  # Assuming closing price for stock price
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        if self.cash_in_hand >= self.stock_price:
            self.available_actions = (0, 1)  # Can hold or buy initially
        else:
            self.available_actions = (0,) ##### Caution: Comma after 0 to make it a tuple maybe delete later
        return self.get_observation()
    
    def step(self, action):
        # assert action in self.available_actions, f'Action {action} not in {self.available_actions} '
        # if action == 2 and self.cash_in_hand < self.stock_price:
        #     action = 0

        prev_valuation = self.total_portfolio_value
        # Execute the chosen action
        if action == 2:  # SELL
            self._sell()
        elif action == 0:  # HOLD
            pass
        elif action == 1:  # BUY
            self._buy()
        # Update the state after executing the action
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        reward = self.total_portfolio_value - prev_valuation   
        done = self.current_step >= self.max_idx

        # Record step data AFTER executing the action and updating the state
        step_data = {
            'Step': self.current_step,
            'Portfolio Value': self.total_portfolio_value,
            'Cash': self.cash_in_hand,
            'Stock Value': self.stock_price * self.stock_holding, 
            'Stock Holdings': self.stock_holding,
            'Stock Price': self.stock_price,
            'Available Actions': self.available_actions,  # This still reflects the previous state
            'Action': action,
            "Reward": reward
        }
        self.step_info.append(step_data)
        
        # Update available actions for the next step
        
        if not done:
            self.current_step += 1
            self.stock_price = self.ohlcv_data[self.current_step][3]

#hid        self.update_available_actions()
        #print("Chosen action:", action, "Available actions before update:", self.available_actions)
        next_observation = self.get_observation()
        info = {'available_actions': self.available_actions}  # This will now reflect the updated state
        return next_observation, reward, done, info

    def _buy(self):
        self.num_stocks_buy = int(np.floor((self.cash_in_hand / self.stock_price) * 1.0))  # Buy up to 100% of stocks possible

        # Check if calculated stocks to buy is zero but cash is enough for at least one stock
        if self.num_stocks_buy == 0 and self.cash_in_hand >= self.stock_price:
            self.num_stocks_buy = 1  # Buy at least one stock

        elif self.num_stocks_buy > 0:
            total_cost = self.num_stocks_buy * self.stock_price
            if total_cost <= self.cash_in_hand:  # Check if we have enough cash
                self.cash_in_hand -= total_cost
                self.stock_holding += self.num_stocks_buy
                self.purchase_history.append((self.current_step + 1, self.stock_price, self.num_stocks_buy))
                self.available_actions = (0, 2)  # Can hold, buy, or sell

    def _sell(self):
        if self.stock_holding > 0:
            self.num_stocks_sell = self.stock_holding
            total_sale_value = self.stock_holding * self.stock_price
            self.cash_in_hand += total_sale_value
            self.stock_holding -= self.num_stocks_sell
            self.available_actions = (0, 1)  # Can hold or buy 
        else:
            # Do nothing if no stocks to sell
            pass


    def get_observation(self): #maybe here different code
        return self.ohlcv_data[self.current_step, :].astype(np.float32)

    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored 
    #step information

#########DQN###########
    
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, epsilon: float =1):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.epsilon = epsilon

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#print(f"{'epoch':<6} {'epsilon':<8} {'total_step':<11} {'final return':<13} {'elapsed_time':<13} {'final portfolio value':<20}")
###TRAIN AND TEST##########

### TRAIN 

def train_dqn(env, ohlcv_data_train,model_name, hidden_size, epoch_num, memory_size, batch_size, train_freq, update_q_freq, gamma, epsilon_decay_divisor, start_reduce_epsilon):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = hidden_size 

    Q = QNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    Q_ast = copy.deepcopy(Q)
    optimizer = optim.Adam(Q.parameters(), lr = 0.001)


    # Hyperparameters
    epoch_num = epoch_num
    step_max = len(ohlcv_data_train[["open", "high", "low", "close", "volume","log_diff"]].to_numpy())-1
    memory_size = memory_size
    batch_size = batch_size
    train_freq = train_freq
    update_q_freq = update_q_freq
    gamma = gamma
    show_log_freq = 1
    epsilon = 1.0
    initial_epsilon = 1.0 #NEW for epsilon decay
    #epsilon_decrease = 1e-3 #NEW Not used for this version, since used epsilon function similar to Mike's
    epsilon_min = 0.05
    epsilon_decay_divisor = epsilon_decay_divisor #NEW - Smaller divisor means slower decay over epochs
    decay_rate = np.log(initial_epsilon/epsilon_min)/(epoch_num/epsilon_decay_divisor)
    start_reduce_epsilon = start_reduce_epsilon  # Start reducing epsilon after this many steps

    memory = []
    total_step = 0
    total_reward = 0
    total_loss = 0
    final_returns = []
    all_epochs_step_data = [] #NEW - Store all step data for all epochs
    epoch_data = []
    performance_data = []
    print(f"{'epoch':<6} {'epsilon':<8} {'total_step':<11} {'final return':<13} {'elapsed_time':<13} {'final portfolio value':<20}")

    start = time.time()
    for epoch in range(epoch_num):
        if epsilon > epsilon_min and total_step > start_reduce_epsilon:
            epsilon = initial_epsilon * np.exp(-decay_rate * epoch) #NEW - Decay epsilon
        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:
            if len(env.step_info) > 0:
                available_actions_choice = env.get_step_data().iloc[-1]["Available Actions"]
            else:
                available_actions_choice = env.available_actions  # Use the initial available actions
                # Epsilon-greedy strategy for action selection
            if np.random.rand() > epsilon:
                # Exploitation: Choose the best action based on Q values
                q_values = Q(torch.from_numpy(np.array(pobs, dtype=np.float32)).unsqueeze(0)).detach()
                #print(f"Q-values: {q_values}")
                # Filter the Q-values based on available actions
                filtered_q_values = {action: q_values[0, action].item() for action in available_actions_choice}
                #print(f"Filtered Q-values: {filtered_q_values}")    
                # Sort the filtered Q-values and select the action with the highest Q-value
                sorted_actions = sorted(filtered_q_values, key=filtered_q_values.get, reverse=True)
                #print(f"Sorted actions: {sorted_actions}")
                # Select the best valid action
                for action in sorted_actions:
                    if action in available_actions_choice:
                        pact = action
                        break
            else:
                # Exploration: Randomly choose from available actions
                pact = random.choice(list(available_actions_choice))

            # Act
            obs, reward, done, _ = env.step(pact)
            #print(f"Step {step}: {env.step_info[-1]}") 

            # Add memory
            memory.append((pobs, pact, reward, obs, done)) ### OBS change to state
            if len(memory) > memory_size:
                memory.pop(0)     

            # Train or update Q
            if len(memory) == memory_size and total_step % train_freq == 0:
                mini_batch = random.sample(memory, batch_size)
                b_pobs = torch.from_numpy(np.array([item[0] for item in mini_batch], dtype=np.float32))
                b_pact = torch.from_numpy(np.array([item[1] for item in mini_batch], dtype=np.int64))
                b_reward = torch.from_numpy(np.array([item[2] for item in mini_batch], dtype=np.float32))
                b_obs = torch.from_numpy(np.array([item[3] for item in mini_batch], dtype=np.float32))
                b_done = torch.from_numpy(np.array([item[4] for item in mini_batch], dtype=bool))

                q = Q(b_pobs)
                maxq = Q_ast(b_obs).detach().max(1)[0]
                target = q.clone()
                for i in range(batch_size):
                    target[i, b_pact[i]] = b_reward[i] + gamma * maxq[i] * (not b_done[i])

                # Loss and optimize
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(q, target)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            # Update Q_ast
            if total_step % update_q_freq == 0:
                Q_ast = copy.deepcopy(Q)
                
            # Next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1



        # Calculate the final portfolio value and return at the end of the epoch
        all_epochs_step_data.extend(env.step_info)
        final_portfolio_value = env.get_step_data().iloc[-1]["Portfolio Value"]
        final_return = int((((final_portfolio_value - env.initial_cash) / env.initial_cash) * 100))
        final_returns.append(final_return)  # Store the return for this epoch
            #FOR WEB
        epoch_data.append({
            'epoch': epoch+1,
            'epsilon': round(epsilon, 4),
            'total_step': total_step,
            'final_return': str(final_return) + '%',
            'elapsed_time': round((time.time() - start), 2),
            'final_portfolio_value': int(final_portfolio_value)})
        # Logging
        if (epoch+1) % show_log_freq == 0:
             elapsed_time = round((time.time() - start), 2)
             print(f"{epoch+1:<6} {round(epsilon, 4):<8} {total_step:<11} {str(final_return) + '%':<14} {elapsed_time:<13} {final_portfolio_value:<20}")
             start = time.time()

        # Convert step_data to DataFrame and export
        all_epochs_step_data_df = pd.DataFrame(all_epochs_step_data)
        all_epochs_filename = f"all_epochs_step_data_{model_name}.csv"
        all_epochs_step_data_df.to_csv(all_epochs_filename, index=False)


    # Convert purchase_history to DataFrame and export
    purchase_history_df = pd.DataFrame(env.purchase_history, columns=["Step Number",'Price', 'Quantity'])
    purchase_history_filename = f"purchase_history_{model_name}.csv"
    purchase_history_df.to_csv(purchase_history_filename, index=True)

    # Final metric calculation
    average_final_return = np.mean(final_returns)
    #print(f'average_final_return: {average_final_return}'+ "%")
    performance_data.append({
            #'avg_reward': average_final_return,
            'avg_final_return': average_final_return})
    torch.save(Q.state_dict(), model_name)
    #model_path = 'trained_model.pth'
    return Q, final_return, final_portfolio_value, average_final_return, final_returns, model_name, epoch_data, performance_data, all_epochs_filename, purchase_history_filename


#### TEST

def test_dqn(env, model_name, ohlcv_data_test, hidden_size, num_episodes=10):

#CAUTION HIDDEN_SIZE NOT SET HERE TO ANYTHING
    
    model = QNetwork(env.observation_space.shape[0], hidden_size, env.action_space.n)
    model.load_state_dict(torch.load(model_name))
    model.eval()  # Set the model to evaluation mode
    epsilon = 0  # Set a fixed epsilon
    total_rewards = []
    step_data_list = []
    step_max = len(ohlcv_data_test[["open", "high", "low", "close", "volume","log_diff"]].to_numpy()) #newwwwwwwwwwwwwwwww
    final_returns_test = []
    num_episodes = num_episodes
    episode_data = []
    total_step = 0
    print(f"{'episode':<6} {'epsilon':<8} {'total_step':<11} {'final return':<13} {'elapsed_time':<13} {'final portfolio value':<20}")

    start = time.time()
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < step_max:
            # Prepare state for model input
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0)

            # Get Q-values from the model for the current state
            with torch.no_grad():
                q_values = model(state_tensor)
            # Get the allowed actions in the current state
            if len(env.step_info) > 0:
                available_actions_choice = env.get_step_data().iloc[-1]["Available Actions"]
            else:
                available_actions_choice = env.available_actions  # Use the initial available actions

            # Choose the action with the highest Q-value among the allowed actions
            if np.random.rand() > epsilon:
                # Exploitation: Choose the best action based on Q values
                filtered_q_values = {action: q_values[0, action].item() for action in available_actions_choice}
                #print(f"filtered q values: {filtered_q_values}")
                sorted_actions = sorted(filtered_q_values, key=filtered_q_values.get, reverse=True)
                #print(f"sorted actions: {sorted_actions}")
                action = next(action for action in sorted_actions if action in available_actions_choice)
                #print(f"action: {action}")
            else:
                # Exploration: Randomly choose from available actions
                action = random.choice(list(available_actions_choice))

            # Execute the chosen action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            portfolio_value = env.total_portfolio_value  # Assuming this attribute exists
            cash = env.cash_in_hand  # Assuming this attribute exists
            stock_value = env.stock_holding * env.stock_price  # Assuming these attributes exist
            stock_holdings = env.stock_holding  # Assuming this attribute exists
            stock_price = env.stock_price  # Assuming this attribute exists
            available_actions = env.available_actions  # Assuming this attribute exists

            # Collect step data for analysis
            step_data = {
                'Episode': episode + 1,
                'Step': env.current_step,
                'Portfolio Value': portfolio_value,
                'Cash': cash,
                'Stock Value': stock_value,
                'Stock Holdings': stock_holdings,
                'Stock Price': stock_price,
                'Available Actions': available_actions,
                'Action': action,
                'Reward': reward,
                'Total Reward': episode_reward,
            }
            
            step_data_list.append(step_data)

            # Update state
            state = next_state
            step += 1
            total_step += 1

        total_rewards.append(episode_reward)
        final_portfolio_value = env.get_step_data().iloc[-1]["Portfolio Value"]
        final_return = round((((final_portfolio_value - env.initial_cash) / env.initial_cash) * 100),2)
        final_returns_test.append(final_return)  # Store the return for this epoch

        # Save step data to a CSV file for analysis
        step_data_df = pd.DataFrame(step_data_list)
        step_data_filename = f"test_step_data_{model_name.rsplit('.', 1)[0]}.csv"
        step_data_df.to_csv(step_data_filename, index=False)
        elapsed_time = round((time.time() - start), 2)
        episode_data.append({
            'episode': episode+1,
            'epsilon': round(epsilon, 4),
            'total_step': total_step,
            'final_return': str(final_return) + '%',
            'elapsed_time': elapsed_time,
            'final_portfolio_value': int(final_portfolio_value)})
        print(f"{episode+1:<6} {round(epsilon, 4):<8} {total_step:<11} {str(final_return) + '%':<14} {elapsed_time:<13} {final_portfolio_value:<20}")
        start = time.time()

    avg_reward = np.mean(total_rewards)
    avg_final_return = np.mean(final_returns_test)
    print(f"Average reward across episodes: {avg_reward} \nAverage final return across episodes: {avg_final_return}")
    return avg_reward,avg_final_return, step_data_list, final_returns_test, episode_data, step_data_filename

# # Define test configurations
# test_runs = {
#     "run_1": {"hidden_size": 10, "epoch_num": 10, "memory_size": 10, "batch_size": 10, "train_freq": 20, "update_q_freq": 50, "gamma": 0.97, "epsilon_decay_divisor": 0.5, "start_reduce_epsilon": 25},
#     "run_2": {"hidden_size": 500, "epoch_num": 10, "memory_size": 300, "batch_size": 40, "train_freq": 400, "update_q_freq": 50, "gamma": 0.97, "epsilon_decay_divisor": 1, "start_reduce_epsilon": 500},
#     "run_3_MADDQN_Inspired":{ "hidden_size": 10,  # Adjust based on the number of linear layers you choose
#     "epoch_num": 10,  # Not mentioned in the paper, keep as is or adjust based on your needs
#     "memory_size": 1000,
#     "batch_size": 10,  # Batch size is not specified in the paper, adjust as needed
#     "train_freq": 10,  # Align with Target Update Frequency
#     "update_q_freq": 10,
#     "gamma": 0.9,
#     "epsilon_decay_divisor": 0.8,
#     "start_reduce_epsilon": 1000}}

def run_many_train_test(test_runs,ohlcv_data_train, ohlcv_data_test):
   
    env_train = ContinuousOHLCVEnv(ohlcv_data_train[["open","high","low",'close',"volume", "log_diff"]].to_numpy())
    env_test = ContinuousOHLCVEnv(ohlcv_data_test[["open","high","low",'close',"volume", "log_diff"]].to_numpy())

    # Assuming you have a way to determine input_size and output_size
    input_size = env_train.observation_space.shape[0] #should not matter if its train or test, bc column number is the same
    output_size = env_train.action_space.n

    results = {}
    data_from_tests = []

    for run_name, params in test_runs.items():

        model = QNetwork(input_size, params['hidden_size'], output_size)
        model_name = f"model_{run_name}.pth"
        Q, final_return, final_portfolio_value, average_final_return, final_returns, model_name, epoch_data, performance_data, all_epochs_filename, purchase_history_filename = train_dqn(env_train, ohlcv_data_train, model_name, **params)
        avg_reward,avg_final_return, step_data_list, final_returns_test, episode_data, step_data_filename = test_dqn(env_test, model_name, ohlcv_data_test, hidden_size = params['hidden_size'])
        data_from_tests.append({
            "run_name": run_name,
            "epoch_data": epoch_data,
            "episode_data": episode_data
        })

        results[run_name] = {
            "parameters": params,
            "train": final_returns,
            "test": final_returns_test
        }
        print(f"Results for {run_name} \nParameters used: {params}\nTraining Returns: {final_returns} \nTesting Returns: {final_returns_test}")
        
    return data_from_tests, results

def plot_stock_data(ohlcv_data, ticker, data_split):
    plt.figure(figsize=(12,6))
    plt.plot(ohlcv_data["date"],ohlcv_data["close"], color='b', linestyle='-')
    plt.title(f"Stock price plot for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")

    split_index = int(len(ohlcv_data)*(data_split/100))
    split_date = ohlcv_data.iloc[split_index]["date"]
    plt.axvline(x=split_date, color="r", linestyle = "--",
                label= "Train/Test data split")
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64_a = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64_a

def plot_model(final_returns, final_returns_test, step_data_filename): #, step_data_filename
    # Ensure that final_returns and final_returns_test are lists
    if not isinstance(final_returns, list):
        final_returns = list(final_returns)
    if not isinstance(final_returns_test, list):
        final_returns_test = list(final_returns_test)

    step_data_df = pd.read_csv(step_data_filename)
    episode_1_data = step_data_df[step_data_df['Episode']==1]

    # Plotting the training and testing results
    plt.figure(figsize=(18, 6))

    # Plotting the training returns
    plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(final_returns, marker='o', color='b', linestyle='-')
    plt.title('Training: Final Return per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Final Return (%)')

    
    # Plotting the testing returns
    plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(final_returns_test, marker='o', color='r', linestyle='-')
    plt.title('Testing: Final Return per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Final Return (End_Portfolio/Start_Cash-Start_Cash)')


    # Plotting buy and sell trades from the testing step data.csv
    plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd subplot

    buy_actions = episode_1_data[episode_1_data['Action']==1]
    sell_actions = episode_1_data[episode_1_data['Action']==2]
    plt.plot(episode_1_data['Step'], episode_1_data['Stock Price'], marker='', color='gray', linestyle='-', label='Price')
    plt.scatter(buy_actions['Step'], buy_actions['Stock Price'],  color='green', label='Buy', s=50)
    plt.scatter(sell_actions['Step'], sell_actions['Stock Price'], color='red', label='Sell', s=50)

    rolling_max = episode_1_data['Portfolio Value'].max()
    drawdown = episode_1_data['Portfolio Value']/rolling_max - 1.0
    max_drawdown = drawdown.min()

    plt.title('Testing trading: Buy and Sell actions')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend(title=f'Trades: {len(buy_actions)} | Max Drawdown: {max_drawdown:.2%}')


    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64

def calculate_trading_statistics(file_path, file_name):
    df = pd.read_csv(file_path)
    df = df[df['Episode'] == 1]

    buy_df = df[df["Action"] == 1]
    sell_df = df[df["Action"] == 2]

    trade_wins = 0
    trade_losses = 0
    trade_returns = []

    # If there are no buy actions, return zeroes
    if buy_df.empty:
        trading_statistics = {
            'file_name': file_name,
            'total_trades': 0,
            'win_percentage': 0,
            'cumulative_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0
        }
        return trading_statistics

    # If there are buys without sells, handle it accordingly
    if len(buy_df) == 1 and len(sell_df) == 0:
        # You could potentially handle unmatched buys here, e.g., by assuming they are held until the end of the period
        last_portfolio_value = df.iloc[-1]['Portfolio Value']
        initial_investment = 10000
        cumulative_return = (last_portfolio_value - initial_investment) / initial_investment

        trading_statistics = {
            'file_name': file_name,
            'total_trades': 0,
            'win_percentage': 0,
            'cumulative_return':cumulative_return, 
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0
        }
        return trading_statistics

    for sel_index, sell_row in sell_df.iterrows():
        sell_price = sell_row["Stock Price"]
        buy_action = buy_df[buy_df["Step"] < sell_row["Step"]].tail(1)
        if not buy_action.empty:
            buy_price = buy_action.iloc[0]["Stock Price"]
            trade_return = (sell_price - buy_price) / buy_price
            trade_returns.append(trade_return)
            if trade_return > 0:
                trade_wins += 1
            else:
                trade_losses += 1

    trade_returns = pd.Series(trade_returns)

    # Handle the case where there are no completed trades
    if not trade_returns.empty or len(buy_df) == (len(sell_df)+1):
        total_trades = len(trade_returns)
        win_percentage = round(trade_wins / total_trades, 2) if total_trades > 1 else 0
        cumulative_return = (trade_returns.add(1).prod() - 1) if total_trades > 1 else 0
        max_drawdown = qs.stats.max_drawdown(trade_returns) if total_trades > 1 else 0
        sharpe_ratio = qs.stats.sharpe(trade_returns) if total_trades > 1 else 0
        sortino_ratio = qs.stats.sortino(trade_returns) if total_trades > 1 else 0
    else:
        total_trades = 0
        win_percentage = 0
        cumulative_return = 0
        max_drawdown = 0
        sharpe_ratio = 0
        sortino_ratio = 0

    trading_statistics = {
        'file_name': file_name,
        'total_trades': total_trades,
        'win_percentage': win_percentage,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio
    }

    return trading_statistics