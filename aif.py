import telebot
from telebot import types
import requests
from bs4 import BeautifulSoup
import investpy
import numpy as np
import socket
from datetime import datetime
from datetime import date
import datetime as dt
import yfinance as yf
import yahoofinancials
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pandas_datareader import data as pdr

plt.style.use('fivethirtyeight')  # специальное отображение графиков для pyplot

token = '2134903896:AAEycNaHkZpqiNq4qVJgcJyMW1-uEVhwIXA'
bot = telebot.TeleBot(token)


# action = input()

@bot.message_handler(content_types=['text'])
def hello(message):
    if message.text == '/start':
        sent = bot.send_message(message.from_user.id, "---🎉AFI Telegram Bot🎉---\nАналитика. \nВведите тикет(можно найти на investing.com)");
        bot.register_next_step_handler(sent, get_name);
    else:
        bot.send_message(message.from_user.id, '/start');


def get_name(message):
    action = message.text
    print(action)
    bot.send_message(message.from_user.id, 'Примерное время выполнения 3 минуты')

    # Основной класс
    class Currency:
        DOLLAR_RUB = 'https://www.google.com/search?sxsrf=ALeKk01NWm6viYijAo3HXYOEQUyDEDtFEw%3A1584716087546&source=hp&ei=N9l0XtDXHs716QTcuaXoAg&q=%D0%B4%D0%BE%D0%BB%D0%BB%D0%B0%D1%80+%D0%BA+%D1%80%D1%83%D0%B1%D0%BB%D1%8E&oq=%D0%B4%D0%BE%D0%BB%D0%BB%D0%B0%D1%80+&gs_l=psy-ab.3.0.35i39i70i258j0i131l4j0j0i131l4.3044.4178..5294...1.0..0.83.544.7......0....1..gws-wiz.......35i39.5QL6Ev1Kfk4'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}

        current_converted_price = 0

        def __init__(self):
            self.current_converted_price = float(self.get_currency_price().replace(",", "."))

        def get_currency_price(self):
            full_page = requests.get(self.DOLLAR_RUB, headers=self.headers)
            soup = BeautifulSoup(full_page.content, 'html.parser')
            convert = soup.findAll("span", {"class": "DFlfde", "class": "SwHCTb", "data-precision": 2})
            return convert[0].text

        def check_currency(self):
            currency = float(self.get_currency_price().replace(",", "."))

            bot.send_message(message.chat.id,"Курс доллара = " + str(currency) + "руб.")

    currency = Currency()
    currency.check_currency()

    class Currency1:
        EURO_RUB = 'https://www.google.com/search?q=курс+евро&sxsrf=AOaemvIHHUkd7Q0URG2r9L94243sihMFzQ%3A1636790156011&source=hp&ei=i2-PYfqAO4iprgSpgJqIDA&iflsig=ALs-wAMAAAAAYY99nHQg3MZXwCxzP1F2CPnoEGNmOFQx&oq=курс+евро&gs_lcp=Cgdnd3Mtd2l6EAMyDQgAEIAEELEDEEYQggIyCwgAEIAEELEDEIMBMggIABCABBCxAzIICAAQgAQQsQMyCwgAEIAEELEDEIMBMgUIABCABDIFCAAQsQMyCwgAEIAEELEDEMkDMggIABCABBCxAzIICAAQgAQQsQM6CAguELEDEIMBOg4ILhCABBCxAxDHARDRAzoICAAQsQMQgwE6EAgAEIAEELEDEIMBEEYQggJQAFiJG2DmH2gAcAB4AIABvwGIAZ0HkgEDNy4ymAEAoAEB&sclient=gws-wiz&ved=0ahUKEwi68sW-7pT0AhWIlIsKHSmABsEQ4dUDCAY&uact=5'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}

        current_converted_price = 0

        def __init__(self):
            self.current_converted_price = float(self.get_currency_price().replace(",", "."))

        def get_currency_price(self):
            full_page = requests.get(self.EURO_RUB, headers=self.headers)
            soup = BeautifulSoup(full_page.content, 'html.parser')
            convert = soup.findAll("span", {"class": "DFlfde", "class": "SwHCTb", "data-precision": 2})
            return convert[0].text

        def check_currency(self):
            currency = float(self.get_currency_price().replace(",", "."))

            bot.send_message(message.chat.id,"Курс евро = " + str(currency) + "руб.")

    currency = Currency1()
    currency.check_currency()

    def Stock_SMA(stock, country):
        ''' stock - stock exchange abbreviation; country - the name of the country'''
        # Read data
        current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)
        try:
            df = investpy.get_stock_historical_data(stock=stock, country=country, from_date='01/01/2019',
                                                    to_date=current_date)
        except:
            df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)
        # Count SMA30 / SMA90
        SMA30 = pd.DataFrame()
        SMA30['Close Price'] = df['Close'].rolling(window=30).mean()
        SMA90 = pd.DataFrame()
        SMA90['Close Price'] = df['Close'].rolling(window=90).mean()
        data = pd.DataFrame()
        data['Stock'] = df['Close']
        data['SMA30'] = SMA30['Close Price']
        data['SMA90'] = SMA90['Close Price']

        # Визуализируем
        plt.figure(figsize=(12.6, 4.6))
        plt.plot(data['Stock'], label=stock, alpha=0.35)
        plt.plot(SMA30['Close Price'], label='SMA30', alpha=0.35)
        plt.plot(SMA90['Close Price'], label='SMA90', alpha=0.35)
        plt.title(stock + ' История (SMA)')
        plt.xlabel('01/01/2019 - ' + current_date)
        plt.ylabel('Цена закрытия')
        plt.legend(loc='upper left')
        plt.savefig('img1.jpg')

    def Stock_EMA(stock, country):
        ''' stock - stock exchange abbreviation; country - the name of the country'''
        # Read data
        current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)
        try:
            df = investpy.get_stock_historical_data(stock=stock, country=country, from_date='01/01/2019',
                                                    to_date=current_date)
        except:
            df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)
        # Count EMA20 / EMA60
        EMA20 = pd.DataFrame()
        EMA20['Close Price'] = df['Close'].ewm(span=20).mean()
        EMA60 = pd.DataFrame()
        EMA60['Close Price'] = df['Close'].ewm(span=60).mean()
        data = pd.DataFrame()
        data['Stock'] = df['Close']
        data['EMA20'] = EMA20['Close Price']
        data['EMA60'] = EMA60['Close Price']

        # Визуализируем
        plt.figure(figsize=(12.6, 4.6))
        plt.plot(data['Stock'], label=stock, alpha=0.35)
        plt.plot(EMA20['Close Price'], label='EMA30', alpha=0.35)
        plt.plot(EMA60['Close Price'], label='EMA60', alpha=0.35)
        plt.title(stock + ' История (EMA)')
        plt.xlabel('01/01/2019 - ' + current_date)
        plt.ylabel('Цена закрытия')
        plt.legend(loc='upper left')
        plt.savefig('img2.jpg')

    def Upper_levels(stock, country):
        current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)
        try:
            df = investpy.get_stock_historical_data(stock=stock, country=country, from_date='01/01/2019',
                                                    to_date=current_date)
        except:
            df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)

        pivots = []
        dates = []
        counter = 0
        lastPivot = 0

        Range = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dateRange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in df.index:
            currentMax = max(Range, default=0)
            value = round(df['High'][i], 2)

            Range = Range[1:9]
            Range.append(value)
            dateRange = dateRange[1:9]
            dateRange.append(i)

            if currentMax == max(Range, default=0):
                counter += 1
            else:
                counter = 0
            if counter == 5:
                lastPivot = currentMax
                dateloc = Range.index(lastPivot)
                lastDate = dateRange[dateloc]
                pivots.append(lastPivot)
                dates.append(lastDate)

        timeD = dt.timedelta(days=30)

        plt.figure(figsize=(12.6, 4.6))
        plt.title(stock + ' История (upper levels)')
        plt.xlabel('01/01/2019 - ' + current_date)
        plt.ylabel('Цена закрытия')
        plt.plot(df['High'], label=stock, alpha=0.35)
        for index in range(len(pivots)):
            plt.plot_date([dates[index], dates[index] + timeD], [pivots[index], pivots[index]], linestyle='-',
                          linewidth=2, marker=",")
        plt.legend(loc='upper left')
        plt.savefig('img3.jpg')

        print('Dates / Prices of pivot points:')
        for index in range(len(pivots)):
            print(str(dates[index].date()) + ': ' + str(pivots[index]))

    def Low_levels(stock, country):
        current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)
        try:
            df = investpy.get_stock_historical_data(stock=stock, country=country, from_date='01/01/2019',
                                                    to_date=current_date)
        except:
            df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)

        pivots = []
        dates = []
        counter = 0
        lastPivot = 0

        Range = [999999] * 10
        dateRange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in df.index:
            currentMin = min(Range, default=0)
            value = round(df['Low'][i], 2)

            Range = Range[1:9]
            Range.append(value)
            dateRange = dateRange[1:9]
            dateRange.append(i)

            if currentMin == min(Range, default=0):
                counter += 1
            else:
                counter = 0
            if counter == 5:
                lastPivot = currentMin
                dateloc = Range.index(lastPivot)
                lastDate = dateRange[dateloc]
                pivots.append(lastPivot)
                dates.append(lastDate)

        timeD = dt.timedelta(days=30)

        plt.figure(figsize=(12.6, 4.6))
        plt.title(stock + ' История (low levels)')
        plt.xlabel('01/01/2019 - ' + current_date)
        plt.ylabel('Цена закрытия')
        plt.plot(df['Low'], label=stock, alpha=0.35)
        for index in range(len(pivots)):
            plt.plot_date([dates[index], dates[index] + timeD], [pivots[index], pivots[index]], linestyle='-',
                          linewidth=2, marker=",")
        plt.legend(loc='upper left')
        plt.savefig('img4.jpg')

        print('Dates / Prices of pivot points:')
        for index in range(len(pivots)):
            print(str(dates[index].date()) + ': ' + str(pivots[index]))

    def Last_Month(stock, country):
        current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)
        try:
            df = investpy.get_stock_historical_data(stock=stock, country=country, from_date='01/01/2019',
                                                    to_date=current_date)
        except:
            df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)
        plt.figure(figsize=(12.6, 4.6))
        plt.plot(df['Close'][-30:], label=stock, alpha=0.35)
        plt.title(stock + ' История за последние 30 дней')
        plt.xlabel('Последние 30 дней')
        plt.ylabel('Цена закрытия')
        plt.legend(loc='upper left')
        plt.savefig('img5.jpg')
        print('Цена за последние пять дней ' + stock + ' =', np.array(df['Close'][-5:][0]), '$;',
              np.array(df['Close'][-5:][1]),
              '$;', np.array(df['Close'][-5:][2]), '$;', np.array(df['Close'][-5:][3]), '$;',
              np.array(df['Close'][-5:][4]), '$;', file=open("output.txt", "a"))
        p_1 = abs(1 - df['Close'][-5:][1] / df['Close'][-5:][0])
        if df['Close'][-5:][1] >= df['Close'][-5:][0]:
            pp_1 = '+' + str(round(p_1 * 100, 2)) + '%'
        else:
            pp_1 = '-' + str(round(p_1 * 100, 2)) + '%'
        p_2 = abs(1 - df['Close'][-5:][2] / df['Close'][-5:][1])
        if df['Close'][-5:][2] >= df['Close'][-5:][1]:
            pp_2 = '+' + str(round(p_2 * 100, 2)) + '%'
        else:
            pp_2 = '-' + str(round(p_2 * 100, 2)) + '%'
        p_3 = abs(1 - df['Close'][-5:][3] / df['Close'][-5:][2])
        if df['Close'][-5:][3] >= df['Close'][-5:][2]:
            pp_3 = '+' + str(round(p_3 * 100, 2)) + '%'
        else:
            pp_3 = '-' + str(round(p_3 * 100, 2)) + '%'
        p_4 = abs(1 - df['Close'][-5:][4] / df['Close'][-5:][3])
        if df['Close'][-5:][4] >= df['Close'][-5:][3]:
            pp_4 = '+' + str(round(p_4 * 100, 2)) + '%'
        else:
            pp_4 = '-' + str(round(p_4 * 100, 2)) + '%'
        print('Процент +/- ' + stock + ' =', pp_1, ';', pp_2, ';', pp_3, ';', pp_4, file=open("output.txt", "a"))

    stock = action

    country = 'NASDAQ'

    Stock_SMA(stock, country)
    Stock_EMA(stock, country)
    Upper_levels(stock, country)
    Low_levels(stock, country)
    Last_Month(stock, country)
    bot.send_photo(message.chat.id, open("img1.jpg", 'rb'))
    bot.send_photo(message.chat.id, open("img2.jpg", 'rb'))
    bot.send_photo(message.chat.id, open("img3.jpg", 'rb'))
    bot.send_photo(message.chat.id, open("img4.jpg", 'rb'))
    bot.send_photo(message.chat.id, open("img5.jpg", 'rb'))

    with open('output.txt', 'r') as f:
        output = f.read()
    bot.send_message(message.chat.id, output)
    path = "output.txt"
    os.remove(path)

    def getData(stocks, start, end):
        stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return returns, meanReturns, covMatrix

    # Portfolio Performance
    def portfolioPerformance(weights, meanReturns, covMatrix, Time):
        returns = np.sum(meanReturns * weights) * Time
        std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
        return returns, std

    stockList = [action]
    stocks = [stock for stock in stockList]
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=360)

    returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
    returns = returns.dropna()

    weights = np.random.random(len(returns.columns))
    weights /= np.sum(weights)

    returns['portfolio'] = returns.dot(weights)

    def historicalVaR(returns, alpha=5):
        """
        Read in a pandas dataframe of returns / a pandas series of returns
        Output the percentile of the distribution at the given alpha confidence level
        """
        if isinstance(returns, pd.Series):
            return np.percentile(returns, alpha)

        # A passed user-defined-function will be passed a Series for evaluation.
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(historicalVaR, alpha=alpha)

        else:
            raise TypeError("Expected returns to be dataframe or series")

        def historicalCVaR(returns, alpha=5):
            """
            Read in a pandas dataframe of returns / a pandas series of returns
            Output the CVaR for dataframe / series
            """
            if isinstance(returns, pd.Series):
                belowVaR = returns <= historicalVaR(returns, alpha=alpha)
                return returns[belowVaR].mean()

            # A passed user-defined-function will be passed a Series for evaluation.
            elif isinstance(returns, pd.DataFrame):
                return returns.aggregate(historicalCVaR, alpha=alpha)

            else:
                raise TypeError("Expected returns to be dataframe or series")

    Time = 1

    VaR = -historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
    CVaR = -historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
    pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

    InitialInvestment = 1000  # ПЕРЕМЕННАЯ
    print('Ожидаемая доходность портфеля в день: ', round(InitialInvestment * pRet, 2), '$', file=open("risk.txt", "a"))
    print('Значение риска 95 CI: ', round(InitialInvestment * VaR, 2), '%', file=open("risk.txt", "a"))
    print('Условный 95CI: ', round(InitialInvestment * CVaR, 2), '%', file=open("risk.txt", "a"))
    with open('risk.txt', 'r') as f:
        risk = f.read()
    bot.send_message(message.chat.id, risk)
    path = "risk.txt"
    os.remove(path)
    bot.send_message(message.chat.id, 'Выполняется анализирование с помощью ИИ')



    AAPL = yf.download(action,
                       start='2020-01-01',
                       end='2021-11-13',
                       progress=False)
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(AAPL['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close price USD')
    plt.savefig('img_ii1.jpg')

    # Создаем новый датафрейм только с колонкой "Close"
    data = AAPL.filter(['Close'])
    # преобразовываем в нумпаевский массив
    dataset = data.values
    # Вытаскиваем количество строк в дате для обучения модели (LSTM)
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data (масштабируем)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Создаем датасет для обучения
    train_data = scaled_data[0:training_data_len]
    # разбиваем на x underscore train и y underscore train
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i])
        y_train.append(train_data[i])

    # Конвертируем x_train и y_train в нумпаевский массив
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # Строим нейронку
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Компилируем модель
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Тренируем модель
    model.fit(x_train, y_train, batch_size=1, epochs=10)

    # Создаем тестовый датасет
    test_data = scaled_data[training_data_len - 60:]
    # по аналогии создаем x_test и y_test
    x_test = []
    y_test = dataset[training_data_len:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i])

    # опять преобразуем в нумпаевский массив
    x_test = np.array(x_test)

    # опять делаем reshape
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Получаем модель предсказывающую значения
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Получим mean squared error (RMSE) - метод наименьших квадратов
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

    # Строим график
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Визуализируем
    plt.figure(figsize=(16, 8))
    plt.title('Модель LSTM')
    plt.xlabel('Дата', fontsize=18)
    plt.ylabel('Цена закрытия', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Pred'], loc='lower right')
    plt.savefig('img_ii1.jpg')
    bot.send_photo(message.chat.id, open("img_ii1.jpg", 'rb'))
    print('Прогноз с использованием нейроных сетей', file=open("ii.txt", "a"))
    with open('ii.txt', 'r') as f:
        ii = f.read()
    bot.send_message(message.chat.id, ii)
    path = "ii.txt"
    os.remove(path)



    tocks = investpy.stocks.get_stocks(country='russia')['symbol']

    counter = 1
    index = 1
    good_stocks = []
    current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)
    a = date.today().month
    if a == 1:
        from_date = str(date.today().day) + '/' + str(12) + '/' + str(date.today().year - 1)
    else:
        from_date = str(date.today().day) + '/' + str(date.today().month - 1) + '/' + str(date.today().year)
    for stock in tocks:
        if counter == 1:
            counter = 0
        try:
            df = investpy.get_stock_historical_data(stock=stock, country='russia', from_date=from_date,
                                                    to_date=current_date)
            technical_indicators = investpy.technical.technical_indicators(stock, 'russia', 'stock',
                                                                           interval='daily')
            country = 'russia'
        except:
            continue
        tech_sell = len(technical_indicators[technical_indicators['signal'] == 'sell'])
        tech_buy = len(technical_indicators[technical_indicators['signal'] == 'buy'])
        moving_averages = investpy.technical.moving_averages(stock, country, 'stock', interval='daily')
        moving_sma_sell = len(moving_averages[moving_averages['sma_signal'] == 'sell'])
        moving_sma_buy = len(moving_averages[moving_averages['sma_signal'] == 'buy'])

        moving_ema_sell = len(moving_averages[moving_averages['ema_signal'] == 'sell'])
        moving_ema_buy = len(moving_averages[moving_averages['ema_signal'] == 'buy'])
        if tech_buy < 9 or tech_sell > 2 or moving_sma_buy < 5 or moving_ema_buy < 5:
            continue
        sma_20 = moving_averages['sma_signal'][2]
        sma_100 = moving_averages['sma_signal'][4]
        ema_20 = moving_averages['ema_signal'][2]
        ema_100 = moving_averages['ema_signal'][4]
        print('Фонд =', stock, file=open("out.txt", "a"))
        print('Технические индикаторы продажи: для покупки =', tech_buy, ', 12; ', 'продавать =', tech_sell, ', 12;',
              file=open("out.txt", "a"))
        print('SMA скользящие средние: покупать =', moving_sma_buy, ', 6; ', 'продавать =', moving_sma_sell, ', 6;',
              file=open("out.txt", "a"))
        print('EMA скользящие средние: покупать =', moving_ema_buy, ', 6; ', 'продавать =', moving_ema_sell, ', 6;',
              file=open("out.txt", "a"))
        print('SMA_20 =', sma_20, ';', 'SMA_100 =', sma_100, ';', 'EMA_20 =', ema_20, ';', 'EMA_100 =', ema_100,
              file=open("out.txt", "a"))
        print('Цены за последние пять дней ' + stock + ' =', np.array(df['Close'][-5:][0]), ';',
              np.array(df['Close'][-5:][1]),
              ';', np.array(df['Close'][-5:][2]), ';', np.array(df['Close'][-5:][3]), ';',
              np.array(df['Close'][-5:][4]), file=open("output.txt", "a"))
        p_1 = abs(1 - df['Close'][-5:][1] / df['Close'][-5:][0])
        if df['Close'][-5:][1] >= df['Close'][-5:][0]:
            pp_1 = '+' + str(round(p_1 * 100, 2)) + '%'
        else:
            pp_1 = '-' + str(round(p_1 * 100, 2)) + '%'
        p_2 = abs(1 - df['Close'][-5:][2] / df['Close'][-5:][1])
        if df['Close'][-5:][2] >= df['Close'][-5:][1]:
            pp_2 = '+' + str(round(p_2 * 100, 2)) + '%'
        else:
            pp_2 = '-' + str(round(p_2 * 100, 2)) + '%'
        p_3 = abs(1 - df['Close'][-5:][3] / df['Close'][-5:][2])
        if df['Close'][-5:][3] >= df['Close'][-5:][2]:
            pp_3 = '+' + str(round(p_3 * 100, 2)) + '%'
        else:
            pp_3 = '-' + str(round(p_3 * 100, 2)) + '%'
        p_4 = abs(1 - df['Close'][-5:][4] / df['Close'][-5:][3])
        if df['Close'][-5:][4] >= df['Close'][-5:][3]:
            pp_4 = '+' + str(round(p_4 * 100, 2)) + '%'
        else:
            pp_4 = '-' + str(round(p_4 * 100, 2)) + '%'
        print('Процент +/- of ' + stock + ' =', pp_1, ';', pp_2, ';', pp_3, ';', pp_4, file=open("out.txt", "a"))
        print()
        with open('out.txt', 'r') as f:
            out = f.read()
        bot.send_message(message.chat.id, out)
        path = "out.txt"
        os.remove(path)

        good_stocks.append(stock)
        break

bot.polling(none_stop=True)
